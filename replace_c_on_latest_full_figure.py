from PIL import Image

base = Image.open('/var/folders/79/jmz351894bl_mz1xycqtssmr0000gn/T/codex-clipboard-c77e92d2-927b-4487-84a5-ae852931ae34.png').convert('RGB')
c = Image.open('/Users/table/TSR/c_bidirectional_scan_ablation_table_replaced_aligned.png').convert('RGB')

# Upper c panel in the supplied full figure. The d panel begins below y=840.
x0, y0, x1, y1 = 0, 0, base.width, 835
c = c.resize((x1-x0, y1-y0), Image.Resampling.LANCZOS)
base.paste(c, (x0, y0))

out = '/Users/table/TSR/figure9_c_replaced_only.png'
base.save(out, dpi=(600, 600))
print(out)
