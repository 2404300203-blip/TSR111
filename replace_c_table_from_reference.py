from PIL import Image

ref = Image.open('/var/folders/79/jmz351894bl_mz1xycqtssmr0000gn/T/codex-clipboard-fc88a494-338d-455f-ac87-f3018430fdc3.png').convert('RGB')
layout = Image.open('/var/folders/79/jmz351894bl_mz1xycqtssmr0000gn/T/codex-clipboard-dbcf08a5-1140-45e8-88d9-f069d6a3b0c8.png').convert('RGB')

# Table-only crop from the real experimental figure.
table = ref.crop((124, 38, 624, 258))

# Exact table slot in the LaTeX-layout image. Only this slot is replaced.
target = (34, 34, 713, 268)
x0, y0, x1, y1 = target
table = table.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
layout.paste(table, (x0, y0))

out = '/Users/table/TSR/c_bidirectional_scan_ablation_table_replaced_aligned.png'
layout.save(out, dpi=(600, 600))
print(out)
