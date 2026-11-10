"""Three vector styles of the SAME diffusion/PPO diagram; preserve all original exports."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib.path import Path as MPath

import build_progress_presentation_assets as base


OUT = base.OUT / 'fig4_style_variants'
DARK = '#19313C'
CAPTURES = {}


def role(color):
    c=str(color).lower()
    if c in {base.BLUE.lower(),base.BLUE_BG.lower()}:return 'blue'
    if c in {base.ORANGE.lower(),base.ORANGE_BG.lower(),'#ffe5ce'}:return 'orange'
    if c in {base.GREEN.lower(),base.GREEN_BG.lower()}:return 'green'
    if c in {base.PURPLE.lower(),base.PURPLE_BG.lower()}:return 'purple'
    return 'neutral'


class ReferenceDiagram(base.Diagram):
    """Reference-like pastel regions, cyan/orange modules and outlined broad flows."""
    def text(self,x,y,s,size=17,*,bold=False,color=base.INK,**kw):
        # Dark typography, as in the supplied reference, instead of colored headings.
        color = DARK if color not in (base.GRAY,) else '#425660'
        return super().text(x,y,s,size,bold=bold,color=color,**kw)

    def box(self,x,y,w,h,fill='white',edge=base.BLUE,*,lw=1.3,dashed=False,radius=13):
        if w>800 and h>180:
            r=role(edge)
            face={'blue':'#DFEBF7','orange':'#FFF5EC','purple':'#F4EEE6'}.get(r,'#F4F7F9')
            border={'blue':'#326984','orange':'#B66732','purple':'#8B7661'}.get(r,DARK)
            return super().box(x,y,w,h,face,border,lw=1.55,dashed=dashed,radius=23)
        if fill in ('#F9FCFE','#F2F8FC'):
            return super().box(x,y,w,h,'#C8E3EF','#7E8D92',lw=1.3,radius=19)
        r=role(edge)
        face={'blue':'#C6E3F0','orange':'#F5CF9E','green':'#F5CF9E','purple':'#E7D9C8'}.get(r,'#C6E3F0')
        if fill=='#FFE5CE':face='#BEE0EE'  # policy loss matches the blue policy module.
        if h<80:face='#B2CEE9' if role(fill)!='green' else '#B7D3C2'
        return super().box(x,y,w,h,face,DARK,lw=max(1.35,lw),dashed=dashed,radius=17 if h>80 else 10)

    def panel(self,x,y,w,h,title,fill,edge):
        self.box(x,y,w,h,fill,edge,lw=1.55,radius=23)
        self.text(x+24,y+28,title,20,bold=True,ha='left')

    def node(self,x,y,w,h,title,lines=(),*,fill='white',edge=base.BLUE,title_size=19,body_size=16):
        if 'Rollout Buffer' not in title:
            return super().node(x,y,w,h,title,lines,fill=fill,edge=edge,
                                title_size=title_size,body_size=body_size)
        # A cylinder echoes the reference's data container; it remains ON-POLICY.
        self.ax.add_patch(Ellipse((x+w/2,y+h-12),w,24,facecolor='#A9D8EB',edgecolor=DARK,lw=1.4,zorder=2))
        self.ax.add_patch(Rectangle((x,y+12),w,h-24,facecolor='#A9D8EB',edgecolor='none',zorder=2))
        self.ax.plot([x,x],[y+12,y+h-12],color=DARK,lw=1.4,zorder=3)
        self.ax.plot([x+w,x+w],[y+12,y+h-12],color=DARK,lw=1.4,zorder=3)
        self.ax.add_patch(Ellipse((x+w/2,y+12),w,24,facecolor='#CEEAF5',edgecolor=DARK,lw=1.4,zorder=3))
        t=self.text(x+w/2,y+43,title,title_size,bold=True);self.text_limits.append((t,w-30))
        for yy,s in zip([y+81,y+112],lines):
            t=self.text(x+w/2,yy,s,body_size);self.text_limits.append((t,w-30))

    def arrow(self,points,*,color=base.BLUE,dashed=False,lw=1.7,head=15):
        r=role(color)
        face={'blue':'#4AA0CC','orange':'#E6AF86','green':'#99CC72','purple':'#A9967B'}.get(r,'#6E9BB3')
        if dashed and r=='orange':face='#71B56A'
        width=(2.4 if dashed else 4.6) if lw>1 else 2.1
        # Smaller encoding arrows keep the masked-chain illustration legible.
        mutation=min(29,head*1.65)
        path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
        patch=FancyArrowPatch(path=path,
            arrowstyle='-|>,head_length=0.7,head_width=0.43',mutation_scale=mutation,
            linewidth=width,color=face,linestyle=(0,(5,3)) if dashed else '-',
            capstyle='round',joinstyle='round',zorder=4)
        patch.set_path_effects([pe.Stroke(linewidth=width+1.5,foreground=DARK),pe.Normal()])
        self.ax.add_patch(patch)

    def label(self,x,y,s,size=14,color=base.GRAY):
        return self.text(x,y,s,size,color=DARK,
            bbox=dict(facecolor='#FFFCF8',edgecolor='none',alpha=.95,pad=2))

    def export(self,name):
        CAPTURES['reference']=[t.get_text() for t in self.text_artists]
        return super().export(name)


class EditorialDiagram(base.Diagram):
    """White editorial cards, navy typography, teal/amber accents and thin routing."""
    NAVY='#233347'
    ACCENTS={'blue':'#167C8C','orange':'#B37F36','green':'#167F70','purple':'#657893','neutral':'#60758A'}

    def text(self,x,y,s,size=17,*,bold=False,color=base.INK,**kw):
        mapped=self.NAVY if bold or color==base.INK else self.ACCENTS.get(role(color),'#5C6C7C')
        return super().text(x,y,s,size,bold=bold,color=mapped,**kw)

    def box(self,x,y,w,h,fill='white',edge=base.BLUE,*,lw=1.3,dashed=False,radius=13):
        if fill in ('#F9FCFE','#F2F8FC'):
            return super().box(x,y,w,h,'#F3F6F8','#D4DEE4',lw=.85,radius=8)
        r=role(edge)
        face='white'
        if h<80:face={'blue':'#F1F6F8','green':'#E4F1EC','purple':'#F0F3F7'}.get(role(fill),'#F1F6F8')
        if role(fill)=='green' and h>80:face='#EEF7F4'
        if fill=='#FFE5CE':face='#FBF6EC'
        return super().box(x,y,w,h,face,'#BFCFD8',lw=1.05,dashed=dashed,radius=8)

    def panel(self,x,y,w,h,title,fill,edge):
        r=role(edge);accent=self.ACCENTS[r]
        super().box(x,y,w,h,'#FAFCFD','#D1DCE3',lw=1,radius=12)
        # An editorial accent rail replaces saturated full-panel backgrounds.
        self.ax.plot([x+22,x+w-22],[y+57,y+57],color='#DEE7EC',lw=1,zorder=3)
        self.ax.plot([x+22,x+112],[y+57,y+57],color=accent,lw=3,zorder=3)
        self.text(x+24,y+28,title,20,bold=True,ha='left')

    def node(self,x,y,w,h,title,lines=(),*,fill='white',edge=base.BLUE,title_size=19,body_size=16):
        at=len(self.text_limits)
        super().node(x,y,w,h,title,lines,fill=fill,edge=edge,title_size=title_size,body_size=body_size)
        for i in range(at,len(self.text_limits)):
            t,_=self.text_limits[i]
            self.text_limits[i]=(t,w-56)
        accent=self.ACCENTS[role(edge)]
        self.ax.plot([x+16,x+16],[y+17,y+h-17],color=accent,lw=2.7,zorder=3,solid_capstyle='round')

    def arrow(self,points,*,color=base.BLUE,dashed=False,lw=1.7,head=15):
        mapped=self.ACCENTS[role(color)] if dashed or role(color)=='green' else '#617A8C'
        super().arrow(points,color=mapped,dashed=dashed,lw=1.5 if dashed else 1.65,head=head)

    def label(self,x,y,s,size=14,color=base.GRAY):
        return self.text(x,y,s,size,color=color,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.97,pad=1.5))

    def export(self,name):
        CAPTURES['editorial']=[t.get_text() for t in self.text_artists]
        return super().export(name)


def svg_text(path):
    root=ET.parse(path)
    return [''.join(t.itertext()) for t in root.findall('.//{http://www.w3.org/2000/svg}text')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    originals={ext:base.OUT/f'fig4_diffusion_training_pipeline.{ext}' for ext in ('svg','pdf','png')}
    before={ext:base.digest(p) for ext,p in originals.items()}
    for ext,p in originals.items():shutil.copy2(p,OUT/f'fig4_v1_original.{ext}')
    old_out=base.OUT
    try:
        base.OUT=OUT
        reference=base.diffusion_training(ReferenceDiagram,'fig4_v2_reference')
        editorial=base.diffusion_training(EditorialDiagram,'fig4_v3_editorial')
    finally:base.OUT=old_out
    assert CAPTURES['reference']==CAPTURES['editorial']
    expected=svg_text(originals['svg'])
    for name in ('fig4_v2_reference','fig4_v3_editorial'):
        assert svg_text(OUT/f'{name}.svg')==expected,name
    assert before=={ext:base.digest(p) for ext,p in originals.items()}
    receipt=dict(original_sha256=before,original_unchanged=True,
        semantic_text_identical=True,module_positions_identical=True,
        text_layout='Reference buffer title/body adjusted within original bounds; editorial padding increased',
        versions=dict(v1='Original blue/orange/purple light panel style',
            v2='Reference-inspired pastel cyan/peach, dark outlines, wide arrows, cylindrical rollout buffer',
            v3='Editorial white cards, navy type, restrained teal/amber accents and thin connectors'),
        formats=['svg','pdf','png'],script_sha256=base.digest(__file__),
        diagram_source_sha256=base.digest(Path(base.__file__)),
        base_diagram_text_count=len(expected),model_loaded=False,training_started=False,
        artifacts=reference+editorial)
    (OUT/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    readme='''# 扩散训练算法图：三版视觉方案

三版的算法内容、公式与模块位置保持一致，仅改变视觉表达，方便选风格。

| 版本 | 视觉特点 | 预览 | 可编辑矢量 |
|---|---|---|---|
| V1 原版 | 浅蓝/浅橙/浅紫分区，细箭头 | [PNG](fig4_v1_original.png) | [SVG](fig4_v1_original.svg) |
| V2 参考图风格 | 接近参考图的蓝底/米橙底、青蓝/杏橙模块、深色描边、宽箭头、柱形Rollout Buffer | [PNG](fig4_v2_reference.png) | [SVG](fig4_v2_reference.svg) |
| V3 简洁编辑风格 | 白色卡片、深蓝文字、克制的青绿/琥珀强调、细连接线 | [PNG](fig4_v3_editorial.png) | [SVG](fig4_v3_editorial.svg) |

每版另有PDF，PNG为240dpi。V1为原文件逐字节复制；原PNG/SVG/PDF未覆盖。三版SVG文本逐元素核对一致。

## V1：原版

![V1 原版](fig4_v1_original.png)

## V2：参考图风格

![V2 参考图风格](fig4_v2_reference.png)

## V3：简洁编辑风格

![V3 简洁编辑风格](fig4_v3_editorial.png)

仅借鉴参考图的视觉语言。仍是本项目拟实施的两阶段扩散/PPO方案：on-policy Rollout Buffer、时间因子critic、团队GAE、逐UAV和逐去噪步PPO；不加入参考论文的回放池、分布型critic、目标网络或熵ELBO。

重建：`.venv/bin/python scripts/build_diffusion_diagram_styles.py`。中文字体为Noto Sans CJK，未安装该字体时可直接使用PNG。
'''
    (OUT/'README.md').write_text(readme)
    archive=OUT/'fig4_three_styles.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    print(f'Three styles ready: {OUT}; original untouched; {len(expected)} SVG text elements identical',flush=True)


if __name__=='__main__':main()
