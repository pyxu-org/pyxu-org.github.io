import os
from collections import defaultdict
from jinja2 import Environment, FileSystemLoader
import sqlite3
import json

DATABASE_FILE = '_parse_plugins/plugins.db'
TEMPLATES_DIR = "_parse_plugins/templates"
OUTPUT_DIR = "pyxu/doc/fair/plugins"
RST_DIR = "rst"

entrypoint_metainfo = {
    "pyxu.operator": {"shortname": "Operator", "colorclass": "blue"},
    "pyxu.opt.solver": {"shortname": "Solver", "colorclass": "brown"},
    "pyxu.opt.stop": {"shortname": "Stop", "colorclass": "purple"},
    "pyxu.math": {"shortname": "Math", "colorclass": "green"},
    "pyxu.contrib": {"shortname": "Contrib", "colorclass": "orange"},
}

status_dict = {
    "1": ["Planning: Not yet ready to use. Developers welcome!", "status-planning-d9644d.svg"],
    "2": ["Pre-alpha: Not yet ready to use. Developers welcome!", "status-planning-d9644d.svg"],
    "3": ["Alpha: Adds new functionality, not yet ready for production. Testing welcome!", "status-alpha-d6af23.svg"],
    "4": ["Beta: Adds new functionality, not yet ready for production. Testing welcome!", "status-beta-d6af23.svg"],
    "5": ["Production/Stable: Ready for production calculations. Bug reports welcome!", "status-stable-4cc61e.svg"],
    "6": ["Mature: Ready for production calculations. Bug reports welcome!", "status-stable-4cc61e.svg"],
    "7": ["Inactive: No longer maintained.", "status-inactive-bbbbbb.svg"],
}

entrypoints_count = defaultdict(list)

def get_summary_info(entry_points):
    summary_info = []
    ep = json.loads(entry_points)

    for entrypoint_name, meta in entrypoint_metainfo.items():
        num = len(ep.get(entrypoint_name, {}))
        if num > 0:
            summary_info.append({"colorclass": meta["colorclass"], "text": meta["shortname"], "count": num})
            entrypoints_count[entrypoint_name].append(num)
    
    return summary_info

def render_plugin_pages(plugins, env):
    if os.path.exists(RST_DIR):
        for f in os.listdir(RST_DIR):
            if f.endswith(".rst"):
                os.remove(os.path.join(RST_DIR, f))
    else:
        os.mkdir(RST_DIR)

    for plugin in plugins:
        summary_info = get_summary_info(plugin["entrypoints"])
        dev_status = status_dict[plugin["development_status"]]
        entry_points = json.loads(plugin["entrypoints"])
        rst_plugin_template = env.get_template("plugin.rst")
        rst_content = rst_plugin_template.render(plugin=plugin, summary_info=summary_info, dev_status=dev_status, entrypointtypes=entrypoint_metainfo, entry_points=entry_points)
        
        with open(os.path.join(OUTPUT_DIR, f'{plugin["name"]}.rst'), 'w') as f:
            f.write(rst_content)

def render_catalogue_page(plugins, plugins_info, env):
    rst_catalogue_template = env.get_template("catalogue.rst")
    rst_content = rst_catalogue_template.render(
        plugins=plugins,
        summary_info=plugins_info["summary_info"],
        dev_status=plugins_info["dev_status"],
        dev_status_count=plugins_info["dev_status_count"],
        summary_info_count=plugins_info["summary_info_count"].values(),
    )

    with open(os.path.join(OUTPUT_DIR, 'index.rst'), 'w') as f:
        f.write(rst_content)

def main():
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT name, pyxu_version, version, author, author_email, home_page, short_description, license, development_status, entrypoints, score FROM plugins ORDER BY name COLLATE NOCASE ASC")
    
    plugins = [{
        'name': row[0],
        'pyxu_version': row[1],
        'version': row[2],
        'author': row[3],
        'author_email': row[4],
        'home_page': row[5],
        'short_description': row[6],
        'license': row[7],
        'development_status': row[8],
        'entrypoints': row[9],
        'score': row[10],
    } for row in c.fetchall()]
    conn.close()

    plugins_info = {
        "summary_info": {},
        "dev_status": {},
        "summary_info_count": {epm["shortname"]: {"colorclass": epm["colorclass"], "num_entries": 0, "name": epm["shortname"], "total_num": 0} for epm in entrypoint_metainfo.values()},
        "dev_status_count": {k: {"badge": v[1], "num_entries": 0} for k, v in status_dict.items()}
    }

    for plugin in plugins:
        summary_info = get_summary_info(plugin["entrypoints"])
        dev_status = status_dict[plugin["development_status"]]
        plugins_info["summary_info"].update({plugin["name"]: summary_info})
        plugins_info["dev_status"].update({plugin["name"]: dev_status})
        
        for entry in summary_info:
            plugins_info["summary_info_count"][entry["text"]]["num_entries"] += 1
            plugins_info["summary_info_count"][entry["text"]]["total_num"] += entry["count"]
        plugins_info["dev_status_count"][plugin["development_status"]]["num_entries"] += 1

    render_plugin_pages(plugins, env)
    render_catalogue_page(plugins, plugins_info, env)

if __name__ == "__main__":
    main()
