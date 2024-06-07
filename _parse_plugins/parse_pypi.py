import sqlite3
import re
import warnings
import requests
from requests.utils import requote_uri
import configparser
import tempfile
import zipfile
import json
from pathlib import Path
from typing import Dict, Tuple, Any

DATABASE_FILE = "_parse_plugins/plugins.db"
TROVE_CLASSIFIER = "Framework :: Pycsou"


def query_pypi() -> Dict[str, str]:
    """
    Query PyPI to get all plugins matching the specified classifier.
    :return: Dictionary with plugin names and their latest versions.
    """
    packages = {}
    name_pattern = re.compile('class="package-snippet__name">(.+?)</span>')
    version_pattern = re.compile('class="package-snippet__version">(.+?)</span>')
    url = requote_uri(f"https://pypi.org/search/?q=&o=-created&c={TROVE_CLASSIFIER}")

    response = requests.get(url)
    response.raise_for_status()
    
    html = response.text
    names = name_pattern.findall(html)
    versions = version_pattern.findall(html)
    
    if len(names) != len(versions):
        return {}
    
    packages = dict(zip(names, versions))
    return packages


class CaseSensitiveConfigParser(configparser.ConfigParser):
    """Case-sensitive config parser."""
    optionxform = staticmethod(str)


def parse_entrypoints(plugin_data: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    build_types = {data.get("packagetype"): data.get("url") for data in plugin_data.get("urls") if data.get("packagetype")}
    
    if "bdist_wheel" not in build_types:
        warnings.warn("No bdist_wheel available for PyPI release")
        return "{}", {}

    wheel_url = build_types.get("bdist_wheel")
    if not wheel_url:
        return "{}", {}

    try:
        with requests.get(wheel_url, stream=True, timeout=120) as download:
            download.raise_for_status()
            with tempfile.TemporaryDirectory() as tmpdirname:
                wheel_path = Path(tmpdirname) / "wheel.whl"
                with wheel_path.open("wb") as handle:
                    for chunk in download.iter_content(chunk_size=8192):
                        handle.write(chunk)
                with zipfile.ZipFile(wheel_path) as whl:
                    entry_points_content = whl.read(next(name for name in whl.namelist() if name.endswith(".dist-info/entry_points.txt"))).decode("utf-8")
                    metadata_content = whl.read(next(name for name in whl.namelist() if name.endswith(".dist-info/METADATA"))).decode("utf-8")

                    # Parse entry points
                    parser = CaseSensitiveConfigParser()
                    parser.read_string(entry_points_content)
                    entry_points = {section: dict(parser.items(section)) for section in parser.sections()}
                    
                    # Parse metadata
                    metadata = parse_metadata(metadata_content)

    except Exception as err:
        warnings.warn(f"Unable to read wheel file from PyPI release of package {plugin_data['info']['name']}: {err}")
        return "{}", {}
    
    return json.dumps(entry_points), metadata


def parse_metadata(metadata_content):
    metadata = {}
    for line in metadata_content.splitlines():
        if line.startswith("Name: "):
            metadata["name"] = line.split("Name: ")[1]
        elif line.startswith("Version: "):
            metadata["version"] = line.split("Version: ")[1]
        elif line.startswith("Author-email: "):
            metadata["author"] = line.split('Author-email: ')[1].split(" <")[0]
        elif line.startswith("Author-email: "):
            metadata["author_email"] = line.split('<')[1][:-1]
        elif line.startswith("Summary: "):
            metadata["short_description"] = line.split("Summary: ")[1]
        elif line.startswith("Project-URL: download, "):
            metadata["home_page"] = line.split("Project-URL: download, ")[1]
        elif line.startswith("License-Expression: "):
            metadata["license"] = line.split("License-Expression: ")[1]
        elif line.startswith("Classifier: Development Status :: "):
            metadata["development_status"] = line.split(":: ")[-1][0]


    return metadata



def store_plugin_data(plugin_data: Dict[str, Any], conn: sqlite3.Connection, c: sqlite3.Cursor):
    entrypoints, metadata = parse_entrypoints(plugin_data)
    if not entrypoints or not metadata:
        return
    name = metadata.get("name", "")
    pyxu_version = "2" # TODO: Change according to min, max pyxu versions
    version = metadata.get("version", "")
    author = metadata.get("author", "")
    author_email = metadata.get("author_email", "")
    home_page = metadata.get("home_page", "")
    short_description = metadata.get("short_description", "")
    license = metadata.get("license", "")
    development_status = metadata.get("development_status", "1")
    score = 100

    c.execute("""
        INSERT INTO plugins 
        (name, pyxu_version, version, author, author_email, home_page, short_description, license, development_status, entrypoints, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, pyxu_version, version, author, author_email, home_page, short_description, license, development_status, entrypoints, score))
    conn.commit()


def main():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS plugins''')
    c.execute('''CREATE TABLE plugins
                 (name TEXT, pyxu_version TEXT, version TEXT, author TEXT, author_email TEXT, home_page TEXT, 
                  short_description TEXT, license TEXT, development_status TEXT, entrypoints TEXT, score INTEGER)''')

    plugin_names = query_pypi()

    for plugin_name, plugin_version in plugin_names.items():
        url = f"https://pypi.org/pypi/{plugin_name}/json"
        response = requests.get(url)
        response.raise_for_status()
        plugin_data = response.json()
        store_plugin_data(plugin_data, conn, c)

    conn.close()

if __name__ == "__main__":
    main()