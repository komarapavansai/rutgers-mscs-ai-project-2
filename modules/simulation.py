"""Deterministic Space Rats engine; policies never receive the target position."""
from collections import deque
from dataclasses import dataclass, field
import copy
import random
import numpy as np

DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def neighbors(grid, p):
    return [(p[0]+r, p[1]+c) for r, c in DIRECTIONS
            if 0 <= p[0]+r < len(grid) and 0 <= p[1]+c < len(grid)
            and grid[p[0]+r, p[1]+c]]


def layout(size, seed):
    rng = random.Random(seed)
    grid = np.zeros((size, size), dtype=bool)
    grid[rng.randrange(1, size-1), rng.randrange(1, size-1)] = True
    while True:
        candidates = [(r,c) for r in range(1,size-1) for c in range(1,size-1)
                      if not grid[r,c] and len(neighbors(grid,(r,c))) == 1]
        if not candidates:
            break
        grid[rng.choice(candidates)] = True
    dead = [tuple(p) for p in np.argwhere(grid) if len(neighbors(grid,tuple(p))) == 1]
    rng.shuffle(dead)
    for r,c in dead[:len(dead)//2]:
        walls = [(r+dr,c+dc) for dr,dc in DIRECTIONS
                 if 0 < r+dr < size-1 and 0 < c+dc < size-1 and not grid[r+dr,c+dc]]
        if walls:
            grid[rng.choice(walls)] = True
    return grid


def normalize(belief):
    total = belief.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Observation has zero probability under the model')
    return belief / total


def predict(grid, belief):
    result = np.zeros_like(belief)
    for p in map(tuple, np.argwhere(grid)):
        dest = neighbors(grid,p) or [p]
        for q in dest:
            result[q] += belief[p] / len(dest)
    return result


def likelihood(shape, position, alpha):
    r,c = np.indices(shape)
    distance = abs(r-position[0]) + abs(c-position[1])
    return np.exp(-alpha*np.maximum(distance-1,0))


def posterior(belief, position, alpha, ping):
    values = likelihood(belief.shape, position, alpha)
    return normalize(belief * (values if ping else 1-values))


def route(grid, start, end):
    queue, prev = deque([start]), {start: None}
    while queue:
        p = queue.popleft()
        if p == end:
            path = []
            while p != start:
                path.append(p)
                p = prev[p]
            return path[::-1]
        for q in neighbors(grid,p):
            if q not in prev:
                prev[q] = p
                queue.append(q)
    return []


def ring_likelihood(grid, position, alpha, beeps, readings):
    """Binomial spatial likelihood, scaled by its maximum for visualization."""
    if not 0 <= beeps <= readings or readings < 1:
        raise ValueError('A batch requires 0 <= beeps <= readings and readings >= 1')
    values = likelihood(grid.shape, position, alpha)
    log_values = np.zeros(grid.shape)
    with np.errstate(divide='ignore'):
        if beeps:
            log_values += beeps * np.log(values)
        if readings-beeps:
            log_values += (readings-beeps) * np.log1p(-values)
    log_values[~grid] = -np.inf
    log_values[position] = -np.inf
    peak = log_values.max()
    if not np.isfinite(peak):
        raise ValueError('Impossible batch under the stationary-target model')
    return np.exp(log_values-peak)


@dataclass
class Agent:
    name: str
    position: tuple
    candidates: set
    belief: np.ndarray
    batch_size: int = 1
    phase: str = 'Localize'
    status: str = 'Searching'
    moves: int = 0
    walls: int = 0
    senses: int = 0
    action: str = 'Ready: position unknown'
    path: list = field(default_factory=list)
    recent_positions: list = field(default_factory=list)
    sense_next: bool = True
    last_ping: str = 'No reading yet'
    finish: int = 0
    displacement: tuple = (0, 0)
    batch_count: int = 0
    beeps: int = 0
    batches: int = 0
    ring: object = None
    previous_ring: object = None
    ring_origin: object = None
    previous_origin: object = None
    escapes: int = 0
    previous_move_success: bool = False


class Simulation:
    def __init__(self, size=30, alpha=.12, seed=7, moving=False, limit=400):
        if not 7 <= size <= 50 or not np.isfinite(alpha) or alpha <= 0 or limit < 1:
            raise ValueError('Use size 7..50, positive finite alpha, and limit >= 1')
        self.size, self.alpha, self.seed, self.moving, self.limit = size, alpha, seed, moving, limit
        self.grid = layout(size,seed)
        cells = list(map(tuple,np.argwhere(self.grid)))
        start,self.rat = random.Random(seed+1).sample(cells,2)
        prior = self.grid.astype(float) / self.grid.sum()
        self.agents = [Agent(name,start,set(cells),prior.copy(),batch_size=n)
                       for name,n in (('Bot 1',1),('Bot 2',20))]
        self.tick = 0
        self.history = []
        self.record()

    def record(self):
        self.history.append(copy.deepcopy((self.tick,self.rat,self.agents)))

    @property
    def done(self):
        return all(a.status != 'Searching' for a in self.agents)

    def blocked(self,p):
        r,c = p
        return sum(not self.grid[r+dr,c+dc] for dr in (-1,0,1)
                   for dc in (-1,0,1) if dr or dc)

    def localize(self,a):
        if a.sense_next:
            count = self.blocked(a.position)
            a.candidates = {p for p in a.candidates if self.blocked(p) == count}
            a.walls += 1
            a.recent_positions.append(a.position)
            a.action = f'Wall sensor: {count} blocked neighbors'
        else:
            scores = [sum((p[0]+dr,p[1]+dc) in neighbors(self.grid,p) for p in a.candidates)
                      for dr,dc in DIRECTIONS]
            best = [i for i,s in enumerate(scores) if s == max(scores)]
            recent = a.recent_positions[-5:]
            wiggle = len(recent) == 5 and (recent.count(recent[-1]) >= 3 or len(set(recent)) <= 2)
            random_direction = a.batch_size == 20 or not a.previous_move_success or wiggle
            if random_direction:
                best = list(range(4))
            if wiggle and a.batch_size == 1:
                a.escapes += 1
            dr,dc = DIRECTIONS[random.Random(self.seed+self.tick*101).choice(best)]
            success = (a.position[0]+dr,a.position[1]+dc) in neighbors(self.grid,a.position)
            a.candidates = {(p[0]+dr,p[1]+dc) if success else p for p in a.candidates
                            if ((p[0]+dr,p[1]+dc) in neighbors(self.grid,p)) == success}
            if success:
                a.position = (a.position[0]+dr,a.position[1]+dc)
                a.displacement = (a.displacement[0]+dr,a.displacement[1]+dc)
            a.previous_move_success = success
            a.moves += 1
            a.action = 'Move in the most commonly open direction' if success else 'Wall bump: rule out locations'
            if random_direction:
                a.action = 'Try a random direction; filter movement outcome'
            if wiggle and a.batch_size == 1:
                a.action = 'Wiggling detected: try a random direction'
        a.sense_next = not a.sense_next
        if len(a.candidates) == 1:
            a.phase, a.sense_next = 'Track', True
            origin = (a.position[0]-a.displacement[0],a.position[1]-a.displacement[1])
            a.belief = self.grid.astype(float)
            a.belief[origin] = 0
            a.belief = normalize(a.belief)
            if self.moving:
                for _ in range(self.tick-1):
                    a.belief = predict(self.grid,a.belief)
            a.action += ' | Located!'

    def contact(self,a):
        """Use the repository's exact co-location check during rat tracking."""
        if a.position == self.rat:
            a.belief[:] = 0
            a.belief[a.position] = 1
            a.last_ping = 'CAPTURE'
            a.status, a.action, a.finish = 'Captured', 'Rat found in the same cell', self.tick
            a.path = []
            return True
        a.belief[a.position] = 0
        a.belief = normalize(a.belief)
        if self.moving and a.ring is not None:
            a.ring[a.position] = 0
            a.ring = normalize(a.ring)
        return False

    def track(self,a,uniform):
        if a.path:
            a.position = a.path.pop(0)
            a.moves += 1
            a.action = f'Follow the selected route: {len(a.path)} steps left'
            self.contact(a)
            if not a.path:
                a.sense_next = True
            return
        if self.contact(a):
            # This branch is only needed when tracking begins on the target.
            a.senses += 1
            return
        if a.sense_next:
            a.batch_count, a.beeps = 0, 0
            a.previous_ring, a.previous_origin = a.ring, a.ring_origin
            a.ring, a.ring_origin = None, a.position
            a.sense_next = False
        a.senses += 1
        ping = uniform < likelihood(self.grid.shape,a.position,self.alpha)[self.rat]
        a.belief = posterior(a.belief,a.position,self.alpha,ping)
        a.last_ping = 'PING' if ping else 'No ping'
        a.batch_count += 1
        a.beeps += int(ping)
        if not self.moving:
            a.ring = ring_likelihood(self.grid,a.position,self.alpha,a.beeps,a.batch_count)
        elif a.batch_size == 20:
            # Display this batch's evidence separately from the accumulated belief.
            if a.ring is None:
                a.ring = self.grid.astype(float)
                a.ring[a.position] = 0
                a.ring = normalize(a.ring)
            a.ring = posterior(a.ring,a.position,self.alpha,ping)
        a.action = f'{a.beeps} beeps in {a.batch_count}/{a.batch_size} readings'
        if a.batch_count == a.batch_size:
            a.batches += 1
            destination = tuple(np.unravel_index(a.belief.argmax(),a.belief.shape))
            a.path = route(self.grid,a.position,destination)
            a.action += '; choose the combined-belief peak'
            if not a.path:
                raise RuntimeError('A nonempty reachable target is required after sensing')

    def step(self):
        if self.done:
            return
        self.tick += 1
        uniform = random.Random(self.seed+100000+self.tick).random()
        for a in self.agents:
            if a.status != 'Searching':
                continue
            if a.phase == 'Localize':
                self.localize(a)
            else:
                self.track(a,uniform)
        if self.moving:
            choices = neighbors(self.grid,self.rat) or [self.rat]
            self.rat = random.Random(self.seed+200000+self.tick).choice(choices)
            for a in self.agents:
                if a.status == 'Searching':
                    a.belief = predict(self.grid,a.belief)
                    if a.ring is not None:
                        a.ring = predict(self.grid,a.ring)
                    if a.phase == 'Track':
                        self.contact(a)
        for a in self.agents:
            if self.tick >= self.limit and a.status == 'Searching':
                a.status, a.finish = 'Budget exhausted', self.tick
                a.action = 'Not captured within the action budget'
        self.record()

    def run(self):
        while not self.done:
            self.step()
        return self

@dataclass(frozen=True)
class ScenarioState:
    agent: Agent
    rat: tuple
    moving: bool

    @property
    def title(self):
        return f'{self.agent.name} / {"Moving" if self.moving else "Stationary"} rat'


class Comparison:
    """Run both target environments together, presenting all four combinations."""
    def __init__(self, size=30, alpha=.12, seed=7, limit=400):
        self.stationary = Simulation(size,alpha,seed,False,limit)
        self.moving = Simulation(size,alpha,seed,True,limit)
        self.size, self.alpha, self.seed, self.limit = size,alpha,seed,limit
        self.grid = self.stationary.grid
        self.tick = 0
        self.history = []
        self.record()

    @property
    def agents(self):
        return [self.stationary.agents[0],self.moving.agents[0],
                self.stationary.agents[1],self.moving.agents[1]]

    @property
    def done(self):
        return self.stationary.done and self.moving.done

    def record(self):
        scenarios = []
        for index in (0,1):
            for world in (self.stationary,self.moving):
                _,rat,agents = world.history[-1]
                scenarios.append(ScenarioState(agents[index],rat,world.moving))
        self.history.append((self.tick,scenarios))

    def step(self):
        if not self.done:
            self.stationary.step()
            self.moving.step()
            self.tick += 1
            self.record()

    def run(self):
        while not self.done:
            self.step()
        return self

    def results(self):
        return [dict(bot=s.agent.name,rat='moving' if s.moving else 'stationary',
                     status=s.agent.status,actions=s.agent.finish,moves=s.agent.moves,
                     sensor_readings=s.agent.senses)
                for s in self.history[-1][1]]
