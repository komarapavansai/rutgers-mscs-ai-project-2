"""Resizable Pygame dashboard and exports from recorded engine states."""
import os
import random
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from pathlib import Path
from PIL import Image
from .simulation import Comparison

BG = (10,17,29)
PANEL = (19,30,46)
TEXT = (233,240,249)
MUTED = (151,171,194)
CYAN = (79,214,233)
GOLD = (255,199,92)
RED = (255,102,119)
GREEN = (102,227,170)
DEFAULT_SPEED = 8


class Dashboard:
    def __init__(self, simulation, size=(1440,1000)):
        pygame.font.init()
        self.sim = simulation
        self.size = size
        self.cursor = 0
        self.playing = False
        self.speed = DEFAULT_SPEED
        self.buttons = []
        self.fonts = {}
        self.icons = {}
        self.map_rects = []

    def text(self,surface,text,x,y,size=18,color=TEXT):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.SysFont('segoeui',size)
        surface.blit(self.fonts[size].render(str(text),True,color),(x,y))

    def step(self):
        if not self.sim.done:
            self.sim.step()
            self.cursor = len(self.sim.history)-1
        if self.sim.done:
            self.playing = False

    def command(self,key):
        if key == 'Initialize':
            s = self.sim
            seed = (s.seed+1+random.SystemRandom().randrange(2147483646)) % 2147483647
            self.sim = Comparison(s.size,s.alpha,seed,s.limit)
            self.cursor = 0
            self.playing = False
        elif key in ('Signal stronger','Signal weaker'):
            old = self.sim
            alpha = round(max(.02,min(2,old.alpha*(.8 if key == 'Signal stronger' else 1.25))),4)
            self.sim = Comparison(old.size,alpha,old.seed,old.limit)
            self.cursor = 0
            self.playing = False
        elif key == 'Start / Stop':
            self.playing = not self.playing if not self.sim.done else False
        elif key == 'Slower':
            self.speed = max(1,self.speed//2)
        elif key == 'Faster':
            self.speed = min(32,self.speed*2)

    def icon(self,surface,center,size,kind):
        size = max(4,int(round(size)))
        key = (kind,size)
        if key not in self.icons:
            art = pygame.Surface((64,64),pygame.SRCALPHA)
            if kind == 'robot':
                pygame.draw.line(art,BG,(32,7),(32,20),7)
                pygame.draw.line(art,CYAN,(32,7),(32,20),4)
                pygame.draw.circle(art,CYAN,(32,6),4)
                pygame.draw.rect(art,BG,(9,17,46,40),border_radius=8)
                pygame.draw.rect(art,CYAN,(12,20,40,34),border_radius=6)
                pygame.draw.rect(art,(10,37,51),(17,27,30,15),border_radius=4)
                pygame.draw.circle(art,TEXT,(24,34),4)
                pygame.draw.circle(art,TEXT,(40,34),4)
                pygame.draw.line(art,BG,(25,47),(39,47),3)
                pygame.draw.rect(art,CYAN,(4,29,6,15),border_radius=2)
                pygame.draw.rect(art,CYAN,(54,29,6,15),border_radius=2)
            else:
                # Side profile: long tail, low body, round ear, and pointed snout.
                pygame.draw.lines(art,BG,False,[(24,43),(12,47),(5,42),(7,34)],6)
                pygame.draw.lines(art,(238,157,158),False,[(24,43),(12,47),(5,42),(7,34)],3)
                pygame.draw.ellipse(art,BG,(18,24,34,27))
                pygame.draw.ellipse(art,(231,217,185),(20,26,30,23))
                pygame.draw.polygon(art,BG,[(41,28),(62,39),(44,49),(35,41)])
                pygame.draw.polygon(art,(239,225,193),[(41,31),(59,39),(44,46),(38,40)])
                pygame.draw.circle(art,BG,(42,24),8)
                pygame.draw.circle(art,(224,195,153),(42,24),6)
                pygame.draw.circle(art,(237,152,155),(42,24),3)
                pygame.draw.circle(art,BG,(49,35),2)
                pygame.draw.circle(art,(246,140,144),(59,39),2)
                pygame.draw.line(art,GOLD,(28,48),(25,52),3)
                pygame.draw.line(art,GOLD,(45,46),(48,50),3)
            bounds = art.get_bounding_rect()
            artwork = art.subsurface(bounds)
            ratio = size/max(bounds.width,bounds.height)
            glyph = pygame.transform.smoothscale(artwork,(max(1,round(bounds.width*ratio)),max(1,round(bounds.height*ratio))))
            self.icons[key] = pygame.Surface((size,size),pygame.SRCALPHA)
            self.icons[key].blit(glyph,((size-glyph.get_width())//2,(size-glyph.get_height())//2))
        surface.blit(self.icons[key],(round(center[0]-size/2),round(center[1]-size/2)))

    def robot(self,surface,center,size=22):
        self.icon(surface,center,size,'robot')

    def rat(self,surface,center,size=22):
        self.icon(surface,center,size,'rat')

    def heatmap(self,surface,values,bx,by,cell,warm=False):
        maximum = max(float(values.max()),1e-15)
        for r in range(self.sim.size):
            for c in range(self.sim.size):
                color = (7,13,22)
                if self.sim.grid[r,c]:
                    t = float(values[r,c])/maximum
                    color = (int(32+26*t),int(49+128*t),int(65+144*t))
                    if warm:
                        color = (int(42+207*t),int(44+48*t),int(52+18*t))
                x,y = round(bx+c*cell),round(by+r*cell)
                width,height = round(bx+(c+1)*cell)-x,round(by+(r+1)*cell)-y
                gap = 1 if cell >= 5 else 0
                pygame.draw.rect(surface,color,(x,y,max(1,width-gap),max(1,height-gap)))

    def ring_story(self,surface,scenario,x,y,width,height):
        a = scenario.agent
        tile = min(int((width-12)/2),int((height-52)/2))
        if tile < 30:
            return
        overlap = a.belief if scenario.moving or a.previous_ring is None or a.ring is None else a.previous_ring*a.ring
        stages = [('1 Previous',a.previous_ring,a.previous_origin,x,y),
                  (f'2 New {a.batch_count}/20',a.ring,a.ring_origin,x+tile+12,y),
                  ('3 Motion + evidence' if scenario.moving else '3 Ring overlap',overlap,None,x+(tile+12)//2,y+tile+28)]
        for index,(label,values,origin,xx,yy) in enumerate(stages):
            self.text(surface,label,xx,yy,11,GOLD if index == 1 else GREEN if index == 2 else MUTED)
            if values is None:
                pygame.draw.rect(surface,BG,(xx,yy+16,tile,tile))
                self.text(surface,'Await sensing',xx+2,yy+tile//2,10,MUTED)
            else:
                self.heatmap(surface,values,xx,yy+16,tile/self.sim.size,True)
                if origin is not None:
                    point = (round(xx+(origin[1]+.5)*tile/self.sim.size),round(yy+16+(origin[0]+.5)*tile/self.sim.size))
                    pygame.draw.circle(surface,CYAN,point,3)
            if index == 2:
                pygame.draw.rect(surface,GREEN,(xx-1,yy+15,tile+2,tile+2),1)
        self.text(surface,'+ ',x+tile+1,y+16+tile//2,15,TEXT)
        note = 'Previous = historical; motion changes overlap' if scenario.moving else 'Green cells on ship: strongest ring overlap'
        self.text(surface,note,x,y+height-10,10,MUTED)

    def render(self):
        w,h = self.size
        surface = pygame.Surface(self.size)
        surface.fill(BG)
        tick,scenarios = self.sim.history[self.cursor]
        self.text(surface,'SPACE RATS',12,6,23)
        self.text(surface,f'Seed {self.sim.seed} | Action {tick}',185,13,14,MUTED)
        self.buttons = []
        x = 12
        for key,width in (('Initialize',175),('Start / Stop',95),('Slower',78),('Faster',78)):
            rect = pygame.Rect(x,39,width,30)
            pygame.draw.rect(surface,(35,52,73),rect,border_radius=5)
            caption = 'Randomize / Initialize' if key == 'Initialize' else ('Stop' if self.playing else 'Start') if key == 'Start / Stop' else key
            self.text(surface,caption,x+10,44,15)
            self.buttons.append((rect,key))
            x += width+7
        pygame.draw.rect(surface,PANEL,(x+3,39,190,30),border_radius=5)
        self.text(surface,f'Speed: {self.speed:d} actions/sec',x+12,44,15,CYAN)
        for key,x in (('Signal stronger',12),('Signal weaker',150)):
            rect = pygame.Rect(x,76,130,28)
            pygame.draw.rect(surface,(35,52,73),rect,border_radius=5)
            self.text(surface,key,x+9,80,14)
            self.buttons.append((rect,key))
        self.text(surface,f'Sensor decay: {self.sim.alpha:.3f} | Changing signal resets this seed',294,82,13,MUTED)
        top,gap,margin = 112,8,8
        card_w = (w-margin*2-gap)//2
        card_h = (h-top-31-gap)//2
        self.panels,self.map_rects = [],[]
        for index,scenario in enumerate(scenarios):
            a = scenario.agent
            left = margin+(index%2)*(card_w+gap)
            y = top+(index//2)*(card_h+gap)
            rect = pygame.Rect(left,y,card_w,card_h)
            self.panels.append((rect,scenario.title))
            pygame.draw.rect(surface,PANEL,rect,border_radius=8)
            status_color = GREEN if a.status == 'Captured' else RED if a.status != 'Searching' else CYAN
            phase = a.status if a.status != 'Searching' else 'Localizing' if a.phase == 'Localize' else 'Moving' if a.path else 'Sensing'
            self.text(surface,scenario.title,left+10,y+6,17)
            self.text(surface,phase,rect.right-(130 if phase == 'Budget exhausted' else 80),y+8,14,status_color)
            counts = a.moves+a.walls+a.senses
            self.text(surface,f'{a.moves} moves | {a.senses} sensor readings | {counts} actions',left+10,y+29,13,MUTED)
            side = min(card_w-16,card_h-87)
            cell = side/self.sim.size
            bx,by = left+(card_w-side)//2,y+50
            if a.batch_size == 20:
                bx = rect.right-side-10
            self.map_rects.append(pygame.Rect(bx,by,side,side))
            values = a.belief.copy()
            if a.phase == 'Localize':
                values[:] = 0
                for pos in a.candidates: values[pos] = 1
            self.heatmap(surface,values,bx,by,cell,a.batch_size == 20 and a.phase == 'Track')
            if a.batch_size == 20 and a.phase == 'Track':
                self.ring_story(surface,scenario,left+10,by,bx-left-20,side)
                if a.previous_ring is not None and a.ring is not None and not scenario.moving and a.status == 'Searching':
                    overlap = a.previous_ring*a.ring
                    peak = overlap.max()
                    if peak > 0:
                        for r,c in np.argwhere(overlap >= peak*.5):
                            pygame.draw.rect(surface,GREEN,(round(bx+c*cell),round(by+r*cell),max(2,round(cell)-1),max(2,round(cell)-1)),1)
            def center(pos):
                return (round(bx+(pos[1]+.5)*cell),round(by+(pos[0]+.5)*cell))
            if a.path:
                points = [center(a.position)]+[center(pos) for pos in a.path]
                pygame.draw.lines(surface,BG,False,points,5)
                pygame.draw.lines(surface,TEXT,False,points,2)
                for j in range(1,len(points),3):
                    tip = pygame.Vector2(points[j])
                    direction = (tip-pygame.Vector2(points[j-1])).normalize()
                    normal = pygame.Vector2(-direction.y,direction.x)
                    base = tip-direction*cell*.45
                    pygame.draw.polygon(surface,TEXT,[tip,base+normal*cell*.25,base-normal*cell*.25])
                target = center(a.path[-1])
                radius = max(4,round(cell*.55))
                pygame.draw.circle(surface,BG,target,radius+2)
                pygame.draw.circle(surface,GREEN,target,radius,2)
                pygame.draw.line(surface,GREEN,(target[0]-radius-2,target[1]),(target[0]+radius+2,target[1]),1)
                pygame.draw.line(surface,GREEN,(target[0],target[1]-radius-2),(target[0],target[1]+radius+2),1)
            if a.status != 'Captured': self.rat(surface,center(scenario.rat),max(14,cell*1.7))
            self.robot(surface,center(a.position),max(14,cell*1.7))
            if a.status == 'Captured':
                r,c = a.position
                pygame.draw.rect(surface,GREEN,(round(bx+c*cell),round(by+r*cell),round(cell),round(cell)),1)
            yy = rect.bottom-33
            if a.status != 'Searching':
                caption = 'Rat captured' if a.status == 'Captured' else 'Action budget reached'
                detail = f'Peak belief: {a.belief.max():.1%}' if a.phase == 'Track' else 'Position was not resolved within the budget'
            elif a.phase == 'Localize':
                caption = f'Finding position: {len(a.candidates)} possible cells'
                detail = 'Cyan cells match the observations so far'
            elif a.batch_size == 20:
                caption = f'Batch: {a.batch_count}/20 sensor readings, {a.beeps} beeps'
                detail = 'Warm cells: accumulated belief + rat motion' if scenario.moving else 'Warm cells: combined probability rings'
            else:
                caption = f'{a.last_ping} | Peak belief: {a.belief.max():.1%}'
                detail = 'One sensor reading, then follow the selected route'
            if a.path and a.status == 'Searching':
                r,c = a.path[-1]
                caption = f'Target: row {r+1}, col {c+1} | {len(a.path)} moves left'
                detail = 'Follow white arrows to the green target'
            self.text(surface,caption,left+10,yy,13,CYAN)
            self.text(surface,detail,left+10,yy+16,12,GOLD if a.batch_size == 20 else MUTED)
        self.robot(surface,(20,h-15),20)
        self.text(surface,'Bot',33,h-23,13,CYAN)
        self.rat(surface,(85,h-15),22)
        self.text(surface,'Rat (viewer only)',99,h-23,13,GOLD)
        self.text(surface,'White: route | Green target: destination | Bot2: previous + new rings = overlap',220,h-23,12,MUTED)
        return surface

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode(self.size,pygame.RESIZABLE)
        pygame.display.set_caption('Space Rats | Four scenarios')
        clock = pygame.time.Clock()
        elapsed = 0
        active = True
        while active:
            dt = clock.tick(60)/1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    active = False
                elif event.type == pygame.VIDEORESIZE:
                    self.size = max(800,event.w),max(720,event.h)
                    screen = pygame.display.set_mode(self.size,pygame.RESIZABLE)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect,key in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.command(key)
                            elapsed = 0
                            break
            if self.playing:
                elapsed += min(dt,.25)
                while self.playing and elapsed >= 1/self.speed:
                    self.step()
                    elapsed -= 1/self.speed
            else:
                elapsed = 0
            screen.blit(self.render(),(0,0))
            pygame.display.flip()
        pygame.quit()

    def export(self,path,stride=1):
        self.sim.run()
        self.speed = DEFAULT_SPEED
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        indices = sorted(set(range(0,len(self.sim.history),stride)) | {len(self.sim.history)-1})
        times = [round(i*100/DEFAULT_SPEED)*10 for i in indices]
        durations = [b-a for a,b in zip(times,times[1:])]+[3000]
        names = ('bot1-stationary','bot1-moving','bot2-stationary','bot2-moving')
        targets = [(path,None)]+[(path.with_name(f'{path.stem}-{name}.gif'),i) for i,name in enumerate(names)]
        for target,panel in targets:
            frames = []
            for i in indices:
                self.cursor = i
                surface = self.render()
                if panel is not None:
                    surface = surface.subsurface(self.panels[panel][0]).copy()
                frames.append(Image.frombytes('RGB',surface.get_size(),pygame.image.tostring(surface,'RGB')).quantize(colors=128))
            frames[0].save(target,save_all=True,append_images=frames[1:],duration=durations,loop=0)
        pygame.image.save(self.render(),str(path.with_suffix('.png')))
