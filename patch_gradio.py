import pathlib

f = pathlib.Path(r"C:\Users\disar\AppData\Local\Programs\Python\Python310\lib\site-packages\gradio_client\utils.py")

txt = f.read_text(encoding="utf-8")

old = 'if "const" in schema:'
new = 'if isinstance(schema, dict) and "const" in schema:'

if old in txt:
    f.write_text(txt.replace(old, new), encoding="utf-8")
    print("PATCHED OK")
else:
    print("Already patched or not found")