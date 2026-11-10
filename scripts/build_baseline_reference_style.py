"""Render the baseline diagram using the same reference-style palette as Fig.4 V2."""
import json

import build_progress_presentation_assets as base
from build_diffusion_diagram_styles import ReferenceDiagram, role, svg_text


class BaselineReferenceDiagram(ReferenceDiagram):
    def panel(self,x,y,w,h,title,fill,edge):
        # Baseline's narrower training panel uses the same pale regional fill.
        r=role(edge)
        face={'blue':'#DFEBF7','orange':'#FFF5EC','purple':'#F4EEE6'}[r]
        border={'blue':'#326984','orange':'#B66732','purple':'#8B7661'}[r]
        base.Diagram.box(self,x,y,w,h,face,border,lw=1.55,radius=23)
        self.text(x+24,y+28,title,20,bold=True,ha='left')


def main():
    originals={ext:base.OUT/f'fig1_baseline_framework.{ext}' for ext in ('svg','pdf','png')}
    hashes={ext:base.digest(path) for ext,path in originals.items()}
    files=base.baseline_framework(BaselineReferenceDiagram,'fig1_baseline_framework_v2_reference')
    assert hashes=={ext:base.digest(path) for ext,path in originals.items()}
    assert svg_text(originals['svg'])==svg_text(base.OUT/'fig1_baseline_framework_v2_reference.svg')
    receipt=dict(original_unchanged=True,original_sha256=hashes,text_and_formulas_identical=True,
        layout_preserved=True,style='Same reference style as Fig.4 V2',artifacts=files,
        script_sha256=base.digest(__file__),model_loaded=False,training_started=False)
    (base.OUT/'fig1_baseline_framework_v2_reference.validation.json').write_text(
        json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print('Baseline V2 generated: SVG / PDF / PNG; original unchanged; identical text and formulas',flush=True)


if __name__=='__main__':main()
