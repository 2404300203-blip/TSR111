from PIL import Image

layout = Image.open('/var/folders/79/jmz351894bl_mz1xycqtssmr0000gn/T/codex-clipboard-afe719f4-2e3c-4de1-b440-155a54c35ba8.png').convert('RGB')
c = Image.open('/Users/table/TSR/c_bidirectional_scan_ablation_table_replaced_aligned.png').convert('RGB')

# c panel occupies the upper-left panel of the supplied manuscript layout.
# Replace only that panel; d and the right-hand manuscript text remain intact.
x0, y0, x1, y1 = 18, 20, 333, 276
scale = min((x1-x0)/c.width, (y1-y0)/c.height)
size = (round(c.width*scale), round(c.height*scale))
resized = c.resize(size, Image.Resampling.LANCZOS)
canvas = Image.new('RGB', (x1-x0, y1-y0), 'white')
canvas.paste(resized, ((canvas.width-size[0])//2, (canvas.height-size[1])//2))
layout.paste(canvas, (x0, y0))

out = '/Users/table/TSR/c_panel_replaced_in_figure9_layout.png'
layout.save(out, dpi=(600,600))
print(out)
