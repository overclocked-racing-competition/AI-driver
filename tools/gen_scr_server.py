# Generate scr_server.xml with N driver slots, all mapped to car1-ow1.
# Note: the compiled scr_server module still caps usable slots at 10.
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
out = sys.argv[2] if len(sys.argv) > 2 else "scr_server.xml"

head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE params SYSTEM "../../libs/tgf/params.dtd">\n'
        '<params name="scr_server" type="robotdef">\n'
        '  <section name="Robots">\n'
        '    <section name="index">\n')
tail = '    </section>\n  </section>\n</params>\n'

body = ""
for i in range(n):
    body += (f'      <section name="{i}">\n'
             f'        <attstr name="name" val="scr_server {i+1}"></attstr>\n'
             f'        <attstr name="desc" val=""></attstr>\n'
             f'        <attstr name="team" val=""></attstr>\n'
             f'        <attstr name="author" val="Daniele Loiacono"></attstr>\n'
             f'        <attstr name="car name" val="car1-ow1"></attstr>\n'
             f'        <attnum name="race number" val="{i+1}"></attnum>\n'
             f'        <attnum name="red" val="1.0"></attnum>\n'
             f'        <attnum name="green" val="1.0"></attnum>\n'
             f'        <attnum name="blue" val="1.0"></attnum>\n'
             f'      </section>\n')

with open(out, "w", encoding="utf-8") as fh:
    fh.write(head + body + tail)
print(f"wrote {out} with {n} indices (all car1-ow1)")
