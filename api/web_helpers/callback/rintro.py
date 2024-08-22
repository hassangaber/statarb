import nbformat
from nbconvert import HTMLExporter


def notebook_to_html(notebook_path:str) -> str:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
    
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'
    
    (body, resources) = html_exporter.from_notebook_node(notebook_content)
    
    # Embed the CSS directly in the HTML
    css = resources['inlining']['css']
    body = f'<style>{css[0]}</style>\n{body}'
    
    return body
