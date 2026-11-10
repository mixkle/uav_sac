"""Build editable paper-style figures from saved results; no model or environment execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/progress_presentation_20261103'
RUN = ROOT / 'runs/mappo_coverage_throughput_capacity10_1m_seed0'
VALIDATION = ROOT / 'analysis/mappo_coverage_throughput_capacity10_1m_seed0'
TOPK = ROOT / 'analysis/topk_capacity10_coverage_throughput_best_diagnostic'
BLUE, BLUE_BG = '#286180', '#EAF3FA'
ORANGE, ORANGE_BG = '#B36B35', '#FFF2E6'
GREEN, GREEN_BG = '#397A68', '#E8F3ED'
PURPLE, PURPLE_BG = '#796C99', '#F0EDF7'
INK, GRAY = '#243540', '#637581'
for font_path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    font_manager.fontManager.addfont(font_path)
FP = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
FB = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({'font.family': FP.get_name(), 'font.size': 12,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False,
                     'savefig.facecolor': 'white'})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def save(fig, name, *, tight=False):
    paths = []
    for ext in ('svg', 'pdf', 'png'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=240, bbox_inches='tight' if tight else None,
                    pad_inches=.08, facecolor='white')
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


class Diagram:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, width), ylim=(height, 0)); self.ax.axis('off')
        self.text_artists = []
        self.text_limits = []

    def text(self, x, y, s, size=17, *, bold=False, color=INK, ha='center', va='center', **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                         fontproperties=FB if bold else FP, zorder=8, **kw)
        self.text_artists.append(t)
        return t

    def box(self, x, y, w, h, fill='white', edge=BLUE, *, lw=1.3, dashed=False, radius=13):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={radius}',
            facecolor=fill, edgecolor=edge, linewidth=lw,
            linestyle=(0, (5, 4)) if dashed else '-', zorder=2)
        self.ax.add_patch(patch)
        return patch

    def panel(self, x, y, w, h, title, fill, edge):
        self.box(x, y, w, h, fill, edge, lw=1.2, radius=18)
        self.text(x + 24, y + 28, title, 20, bold=True, color=edge, ha='left')

    def node(self, x, y, w, h, title, lines=(), *, fill='white', edge=BLUE,
             title_size=19, body_size=16):
        self.box(x, y, w, h, fill, edge)
        t=self.text(x + w / 2, y + 30, title, title_size, bold=True)
        self.text_limits.append((t,w-24))
        if lines:
            start,end=(58,h-15) if h<120 else (66,h-24)
            ys = np.linspace(y+start,y+end,len(lines)) if len(lines)>1 else [y+h*.67]
            for at, label in zip(ys, lines):
                t=self.text(x+w/2,at,label,body_size)
                self.text_limits.append((t,w-24))

    def arrow(self, points, *, color=BLUE, dashed=False, lw=1.7, head=15):
        path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
        patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
            linewidth=lw, linestyle=(0, (5, 3)) if dashed else '-', color=color,
            capstyle='round', joinstyle='round', zorder=4)
        self.ax.add_patch(patch)

    def label(self, x, y, s, size=14, color=GRAY):
        return self.text(x, y, s, size, color=color,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=2))

    def export(self, name):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for t,limit in self.text_limits:
            width=t.get_window_extent(renderer).width
            pixel_limit=limit/self.w*self.fig.bbox.width
            if width>pixel_limit:
                t.set_fontsize(t.get_fontsize()*pixel_limit/width)
        self.fig.canvas.draw()
        bounds = self.fig.bbox
        for t in self.text_artists:
            r = t.get_window_extent(self.fig.canvas.get_renderer())
            if r.x0 < bounds.x0 - 2 or r.x1 > bounds.x1 + 2 or r.y0 < bounds.y0 - 2 or r.y1 > bounds.y1 + 2:
                raise ValueError(f'Figure text outside canvas: {t.get_text()}')
        return save(self.fig, name)


def baseline_framework(diagram_class=Diagram, output_name='fig1_baseline_framework'):
    d = diagram_class(1600, 650)
    d.panel(25, 35, 1000, 545, '(a) 分布执行与环境交互', BLUE_BG, BLUE)
    d.panel(1055, 35, 520, 545, '(b) 集中式训练', ORANGE_BG, ORANGE)
    d.node(65, 140, 240, 135, '局部观察', ['自身 / BS / 邻居', '感知UE集合＋一步历史'])
    d.node(360, 140, 270, 135, '共享 Gaussian Actor',
           [r'$z_i\sim\mathcal{N}(\mu_i,\sigma_i^2)$', '圆盘映射 → 连续运动'])
    d.node(685, 140, 290, 135, '运动指令',
           [r'$a_i=(a_{i,x},a_{i,y})$', r'$\|\Delta p_i\|\leq30\;\mathrm{m}$'])
    d.arrow([(305, 207), (360, 207)]); d.arrow([(630, 207), (685, 207)])
    d.node(515, 385, 460, 145, '环境：运动与集中式UE关联',
           ['UAV / UE运动 → 无线与回传更新', '集中式规则确定UE关联 → 团队奖励'])
    d.arrow([(830, 275), (830, 385)])
    d.label(830, 328, '联合运动指令', 15)
    d.arrow([(515, 460), (185, 460), (185, 275)])
    d.label(320, 442, r'下一时刻观察 $o_i^{t+1}$', 16)
    d.node(1100, 140, 430, 140, 'GAE ＋ PPO更新',
           [r'团队优势 $\hat A_t$ → 更新共享actor', '价值目标 → 更新critic'], edge=ORANGE)
    d.node(1100, 385, 430, 145, '集中式时间因子 Critic',
           [r'$V_\omega(s_t)=H_t\,\hat m_\omega(s_t)$', '全局状态仅在训练阶段使用'],
           fill=GREEN_BG, edge=GREEN)
    d.arrow([(975, 457), (1100, 457)], color=GREEN)
    d.label(1035, 435, r'$s_t,r_t,d_t$', 14, GREEN)
    d.arrow([(1230, 385), (1230, 280)], color=GREEN)
    d.label(1230, 331, '累计价值', 15, GREEN)
    d.arrow([(1450, 280), (1450, 385)], color=ORANGE, dashed=True)
    d.label(1450, 331, '价值回归', 15, ORANGE)
    d.arrow([(1100, 205), (1040, 205), (1040, 105), (495, 105), (495, 140)],
            color=ORANGE, dashed=True)
    d.label(760, 103, '策略参数更新', 15, ORANGE)
    d.text(800, 615, r'容量 $N=10$   ·   团队奖励 $r_t=\rho_t+0.1\tau_t$   ·   共享连续MAPPO / CTDE', 18)
    return d.export(output_name)


def diffusion_training(diagram_class=Diagram, output_name='fig4_diffusion_training_pipeline'):
    d = diagram_class(1800, 1020)
    d.panel(30, 25, 1740, 190, '(a) 阶段 I：联合动作示范与去噪预训练', PURPLE_BG, PURPLE)
    d.node(60, 90, 340, 100, '示范教师', ['冻结运动策略＋局部SINR评分'], edge=PURPLE, body_size=15)
    d.node(455, 90, 345, 100, '联合示范数据',
           [r'$(o_i,z_i^*,\mathrm{mask}_i)$'], edge=PURPLE)
    d.node(855, 90, 440, 100, '加噪 → 条件去噪预训练',
           ['联合训练集合编码器与去噪器'], edge=PURPLE, body_size=16)
    d.node(1350, 90, 385, 100, '初始化在线 Actor',
           [r'参数 $\Theta=(\phi,\psi,\theta)$'], edge=PURPLE)
    for a, b in [(400,455),(800,855),(1295,1350)]: d.arrow([(a,140),(b,140)],color=PURPLE)

    d.panel(30, 260, 860, 680, '(b) 分布执行：on-policy采集', BLUE_BG, BLUE)
    d.panel(925, 260, 845, 680, '(c) 集中训练：时间因子 Critic ＋ PPO', ORANGE_BG, ORANGE)
    d.node(80, 330, 440, 135, '本机局部观察',
           ['固定特征＋历史＋变长感知UE集合', r'可服务mask：$\mathcal{F}_i^t\subseteq\mathcal{D}_i^t$'],
           body_size=16)
    d.node(610, 330, 245, 155, '环境与采集器',
           ['① t时刻Top-K提议', '② UAV / UE运动', '③ t+1门控与UE选择'],
           title_size=18, body_size=15)
    d.arrow([(610, 400),(520, 400)])
    d.label(566, 376, r'$o_i^{t+1}$', 15)

    # Stacked cards indicate one shared policy applied independently per UAV.
    d.box(100, 485, 700, 220, '#F9FCFE', '#A8BEC9')
    d.box(90, 495, 700, 220, '#F2F8FC', '#8CAEBF')
    d.box(80, 505, 700, 220, 'white', BLUE, lw=1.6)
    d.text(430, 530, '共享集合条件化 Diffusion Actor', 21, bold=True)
    d.box(105, 555, 650, 44, BLUE_BG, BLUE, radius=8)
    d.text(430, 577, 'UE集合编码＋上下文融合＋有效坐标mask', 16)
    xs=[110,285,460,635]
    for x, title, fill in zip(xs, [r'$z_i^{(L)}$',r'$z_i^{(L-1)}$',r'$\cdots$',r'$z_i^{(0)}$'],
                              [PURPLE_BG,BLUE_BG,BLUE_BG,GREEN_BG]):
        d.box(x, 625, 115, 66, fill, BLUE, radius=8)
        d.text(x+57.5, 655, title, 21)
        d.arrow([(x+57.5,599),(x+57.5,625)],head=10,lw=1)
    for a,b in zip(xs[:-1],xs[1:]):d.arrow([(a+115,658),(b,658)],head=12)
    d.text(430, 707, r'逐UAV采样：2维运动＋$|\mathcal{F}_i^t|$个评分latent', 15)
    d.arrow([(300,465),(300,505)])
    d.node(590, 775, 265, 105, '联合动作映射',
           ['运动：圆盘映射', 'UE评分：Sigmoid'], title_size=18, body_size=15)
    d.arrow([(710,725),(710,775)])
    d.arrow([(855,827),(875,827),(875,410),(855,410)])
    d.label(871, 751, r'$a_i^t$', 15)
    d.arrow([(1542,190),(1542,235),(45,235),(45,570),(80,570)],
            color=PURPLE,dashed=True,lw=1.4)
    d.label(960,235,'预训练参数初始化；示范数据不混入PPO概率比',14,PURPLE)

    d.node(980, 330, 730, 135, '本轮 Rollout Buffer（无离策略回放池）',
           [r'$o_t,s_t,r_t,d_t,\mathrm{mask},\{z^{(L)},\ldots,z^{(0)}\}$',
            r'采样时的 $\log p_{\Theta_{old}}$ 与旧累计价值'],
           title_size=19, body_size=16, edge=ORANGE)
    d.arrow([(735,330),(735,318),(1350,318),(1350,330)], color=BLUE)
    d.label(1095,318,'交互轨迹＋采样器保存的去噪链',12)

    d.node(980, 525, 320, 155, '时间因子 Critic',
           [r'$V_\omega(s_t)=H_t\hat m_\omega(s_t)$', r'$H_t=(1-\gamma^{T-t})/(1-\gamma)$',
            '输入全局状态；输出累计价值'], edge=GREEN, fill=GREEN_BG, body_size=15)
    d.node(1390, 525, 320, 155, '团队 GAE 与回报目标',
           [r'$\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t$',
            r'$\hat A_t\ \ ;\ \ \hat R_t=\hat A_t+V_{old}(s_t)$',
            '真终止不bootstrap'], edge=ORANGE, body_size=14)
    d.arrow([(1140,465),(1140,525)],color=GREEN)
    d.label(1140,496,r'$s_t,s_{t+1}$',14,GREEN)
    d.arrow([(1550,465),(1550,525)],color=ORANGE)
    d.label(1550,496,r'$r_t,d_t$',14,ORANGE)
    d.arrow([(1300,603),(1390,603)],color=GREEN)
    d.label(1345,580,'V值',14,GREEN)

    d.node(980, 780, 320, 115, 'Critic价值回归',
           ['累计价值空间 Huber＋Value Clip', '平均奖励PopArt / 独立优化器'],
           edge=GREEN, fill=GREEN_BG, title_size=18, body_size=14)
    d.node(1390, 780, 320, 115, '去噪转移 PPO Clip',
           ['逐UAV、逐随机去噪步计算概率比', '在同一条旧去噪链上重算概率'],
           edge=ORANGE, fill='#FFE5CE', title_size=18, body_size=14)
    d.arrow([(1490,680),(1490,730),(1140,730),(1140,780)],color=GREEN)
    d.label(1310,730,r'累计价值目标 $\hat R_t$',14,GREEN)
    d.arrow([(1640,680),(1640,780)],color=ORANGE)
    d.label(1640,735,r'团队优势 $\hat A_t$',14,ORANGE)
    d.arrow([(1710,400),(1735,400),(1735,835),(1710,835)],color=ORANGE)
    d.label(1738,708,'同链评估',13,ORANGE)
    d.arrow([(980,840),(950,840),(950,650),(980,650)],color=GREEN,dashed=True)
    d.arrow([(1540,895),(1540,920),(430,920),(430,725)],color=ORANGE,dashed=True,lw=1.8)
    d.label(950,919,'策略梯度更新集合编码器、上下文与去噪器；不经过Top-K反向传播',14,ORANGE)
    d.text(900,980,r'拟实施方案｜容量10，$r_t=\rho_t+0.1\tau_t$｜实线：数据 / 价值；虚线：参数初始化或更新',17)
    return d.export(output_name)


def chart_style():
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.15,'axes.labelsize':12,
                         'axes.titlesize':14,'legend.frameon':False})


def baseline_charts():
    chart_style()
    with (RUN/'validation.csv').open() as f: rows=list(csv.DictReader(f))
    x=np.array([int(r['env_steps']) for r in rows]); best_idx=int(np.argmax([float(r['return']) for r in rows]))
    fig,axes=plt.subplots(1,3,figsize=(13.6,3.65))
    for ax,key,title,scale in zip(axes,['coverage','throughput_bps','return'],
            ['验证平均覆盖率','验证平均总吞吐','验证完整回合回报'],[100,1e-6,1]):
        y=np.array([float(r[key]) for r in rows])*scale
        ax.plot(x/1000,y,color=BLUE,lw=2)
        ax.scatter(x[best_idx]/1000,y[best_idx],color=ORANGE,s=70,zorder=5,marker='*',label='best：671,744步')
        ax.scatter(x[-1]/1000,y[-1],facecolor='white',edgecolor=BLUE,s=38,zorder=5,label='last：1M步')
        ax.set(title=title,xlabel='联合转移数（k）',ylabel={'coverage':'覆盖率（%）','throughput_bps':'Mbps','return':'回合回报'}[key])
    axes[0].legend(fontsize=9,loc='lower right')
    fig.suptitle('容量10基准｜32个固定验证场景｜采样策略｜单训练种子',fontsize=13,y=1.01)
    fig.tight_layout(); paths=save(fig,'fig2_baseline_validation_progress',tight=True)
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.5))
    for name,color in [('best',BLUE),('last',ORANGE)]:
        with np.load(VALIDATION/f'validation_{name}'/'reward_curves.npz') as d:
            names=list(d['diagnostic_names']);diag=d['diagnostics'].mean(0)
            for ax,key,title in zip(axes,['coverage','disconnection_ratio'],['回合内覆盖维持','回合内UAV断联比例']):
                ax.plot(d['steps'],diag[:,names.index(key)]*100,label=name,color=color,lw=1.7)
                ax.set(title=title,xlabel='回合步数',ylabel='比例（%）');ax.legend(fontsize=11)
    fig.tight_layout();paths+=save(fig,'fig2_episode_maintenance',tight=True)
    summary=json.loads((VALIDATION/'report.json').read_text())
    table=[]
    table.append(dict(model='initial',env_steps=0,coverage_pct=float(rows[0]['coverage'])*100,
        throughput_mbps=float(rows[0]['throughput_bps'])/1e6,episode_return=float(rows[0]['return']),
        last100_coverage_pct='',final_coverage_pct=''))
    for name in ['best','last']:
        item=summary['evaluations'][name];m=item['summary']['mean']
        table.append(dict(model=name,env_steps=item['checkpoint_env_steps'],coverage_pct=m['coverage']*100,
            throughput_mbps=m['throughput_bps']/1e6,episode_return=m['return'],
            last100_coverage_pct=m['last100_coverage']*100,final_coverage_pct=m['final_coverage']*100))
    with (OUT/'table2_baseline_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    return paths


def topology_snapshot():
    chart_style()
    with np.load(VALIDATION/'validation_best/trace_10000.npz') as d:
        xy,ue,bs,A,B,C=[d[k][-1] if k!='bs_positions' else d[k] for k in ['uav_xy','user_xy','bs_positions','A','B','C']]
    fig,ax=plt.subplots(figsize=(5.7,5.6))
    reachable=(B<=len(xy)).any(-1);served=C.any(0)
    for i,j in zip(*np.where(np.triu(A,1))):ax.plot(*xy[[i,j]].T,color=BLUE,lw=1,alpha=.65,zorder=1)
    for i in range(len(xy)):
        for g in range(len(bs)):
            if B[i,g]==1:ax.plot([xy[i,0],bs[g,0]],[xy[i,1],bs[g,1]],color=ORANGE,lw=1,ls='--',alpha=.8,zorder=1)
    ax.scatter(*ue[~served].T,s=16,color='#B1B9BE',label='未服务UE',zorder=2)
    ax.scatter(*ue[served].T,s=17,color=GREEN,label='已服务UE',zorder=3)
    ax.scatter(*xy[reachable].T,s=63,marker='^',color=BLUE,edgecolor='white',lw=.5,label='联网UAV',zorder=4)
    if (~reachable).any():ax.scatter(*xy[~reachable].T,s=63,marker='^',color=ORANGE,label='断联UAV',zorder=4)
    ax.scatter(bs[:,0],bs[:,1],s=180,marker='*',color=ORANGE,label='BS',zorder=5)
    for i,p in enumerate(xy):ax.annotate(str(i+1),p,xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlim=(-80,3580),ylim=(-80,3580),aspect='equal',xlabel='x（m）',ylabel='y（m）',
           title=f'新基准best｜场景10000，t=400\n覆盖{served.mean():.0%}；联网UAV {reachable.sum()}/18')
    ax.legend(loc='upper left',fontsize=9,ncol=2)
    fig.tight_layout()
    return save(fig,'fig1_baseline_topology_scene10000_t400',tight=True)


def topk_results():
    chart_style()
    s=json.loads((TOPK/'summary.json').read_text());table=[]
    for g in ['teacher','constant','random']:
        p=s['groups'][g]['P0']['all'];p1=s['groups'][g]['P1']['all'];at=lambda k:p[k]['mean']
        table.append(dict(scoring=g,P0_coverage_pct=at('coverage')*100,P1_coverage_pct=p1['coverage']['mean']*100,
            O1_coverage_pct=(at('coverage')+at('protocol_loss')/150)*100,
            O2_coverage_pct=(at('coverage')+(at('protocol_loss')+at('timing_loss'))/150)*100,
            P0_throughput_mbps=at('throughput_bps')/1e6,P1_throughput_mbps=p1['throughput_bps']['mean']/1e6,
            P0_task_reward=at('task_reward'),P1_task_reward=p1['task_reward']['mean']))
    with (OUT/'table5_topk_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    teacher=table[0]
    heights=[teacher[f'{name}_coverage_pct'] for name in ['P0','P1','O1','O2']]
    bars=axes[0].bar(range(4),heights,color=[BLUE,'#82AFC5',GREEN,ORANGE],width=.62)
    for b,h in zip(bars,heights):axes[0].text(b.get_x()+b.get_width()/2,h+.4,f'{h:.2f}%',ha='center',fontsize=11)
    axes[0].set(xticks=range(4),xticklabels=['P0\n一轮Top-K','P1\n补位','O1\n同信息上界','O2\n动作后上界'],
                ylim=(0,67),ylabel='平均覆盖率（%）',title='SINR评分：协议与时序机会')
    rates=[r['P0_throughput_mbps'] for r in table]
    bars=axes[1].bar(range(3),rates,color=[BLUE,'#A6B1BC','#D4DBDF'],width=.6)
    for b,h in zip(bars,rates):axes[1].text(b.get_x()+b.get_width()/2,h+4,f'{h:.2f}',ha='center',fontsize=11)
    axes[1].set(xticks=range(3),xticklabels=['SINR评分','常数评分','随机评分'],ylim=(0,550),
                ylabel='Mbps',title='一轮Top-K：评分对吞吐的影响')
    fig.tight_layout()
    return save(fig,'fig5_topk_summary',tight=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=[RUN/'validation.csv',VALIDATION/'report.json',TOPK/'summary.json',
            VALIDATION/'validation_best/trace_10000.npz']
    source += [VALIDATION/f'validation_{name}'/'reward_curves.npz' for name in ['best','last']]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in source}
    files=baseline_framework()+diffusion_training()+baseline_charts()+topology_snapshot()+topk_results()
    assert hashes=={str(p.relative_to(ROOT)):digest(p) for p in source}
    record=dict(artifacts=files,source_sha256=hashes,
        algorithm_status='Baseline trained; diffusion DPPO training diagram is a proposed design',
        style_reference='Diffusing to Coordinate Efficient Online Multi-Agent Diffusion Policies.pdf, Figure 1, PDF page 5; visual organization only',
        diagram_implementation='Matplotlib vector drawing; text retained in SVG, embedded fonts in PDF',
        data_resampled=False,model_loaded=False,training_started=False,
        script_sha256=digest(__file__))
    (OUT/'assets_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(f'Generated {len(files)} figures in {OUT}',flush=True)


if __name__=='__main__':main()
