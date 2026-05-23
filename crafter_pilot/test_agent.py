import crafter
import numpy as np
import sys
sys.path.insert(0, '.')
from heuristic_agent import CrafterHeuristicAgent

agent = CrafterHeuristicAgent(seed=42)
results = []
for ep in range(5):
    env = crafter.Env(size=(64,64), seed=100+ep)
    obs = env.reset()
    # Get initial info via NOOP step
    obs, _, done, info = env.step(0)
    if done:
        results.append((ep, 1, 0, 'DEAD'))
        continue
    agent.reset()
    total_reward = 0
    prev_health = info.get('inventory', {}).get('health', 9)
    damage_log = []
    for step in range(1, 10000):
        action = agent.act(obs, info)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        cur_health = info.get('inventory', {}).get('health', 0)
        if cur_health < prev_health:
            px, py = int(info['player_pos'][0]), int(info['player_pos'][1])
            sem = info['semantic']
            nearby = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = px+dx, py+dy
                    if 0<=nx<64 and 0<=ny<64:
                        v = int(sem[nx,ny])
                        if v in (14,15,16,17):
                            dist = abs(dx)+abs(dy)
                            names = {14:'cow',15:'zombie',16:'skeleton',17:'arrow'}
                            nearby.append(names[v] + '@d' + str(dist))
            dmg = prev_health - cur_health
            msg = '  step %d: hp %d->%d (%d dmg) nearby=[%s]' % (step, prev_health, cur_health, dmg, ','.join(nearby))
            damage_log.append(msg)
        prev_health = cur_health
        if done:
            break
    status = 'DEAD' if done else 'SURVIVED'
    results.append((ep, step+1, total_reward, status))
    print('Episode %d (seed=%d): %s at step %d, reward=%.1f' % (ep, 100+ep, status, step+1, total_reward))
    if damage_log:
        print('  Damage events (%d):' % len(damage_log))
        for line in damage_log[:20]:
            print(line)
        if len(damage_log) > 20:
            print('  ... and %d more' % (len(damage_log)-20))
    inv_f = info.get('inventory', {})
    print('  Final inv: health=%d food=%d drink=%d energy=%d wood=%d' % (
        inv_f.get('health',0), inv_f.get('food',0), inv_f.get('drink',0),
        inv_f.get('energy',0), inv_f.get('wood',0)))
    print('  Weapons: wood_sword=%d stone_sword=%d wood_pickaxe=%d' % (
        inv_f.get('wood_sword',0), inv_f.get('stone_sword',0), inv_f.get('wood_pickaxe',0)))
    print()

deaths = sum(1 for _,_,_,s in results if s=='DEAD')
avg_steps = np.mean([s for _,s,_,_ in results])
avg_reward = np.mean([r for _,_,r,_ in results])
print('Summary: %d/5 deaths (%d%%), avg_steps=%d, avg_reward=%.1f' % (deaths, deaths*100//5, int(avg_steps), avg_reward))
