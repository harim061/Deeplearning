import numpy as np
arms_profit=[0.4,0.12,0.52,0.6,0.25]
n_arms =len(arms_profit)

n_trial = 10000

def pull_bandit(handle):
    q=np.random.random()
    if q<arms_profit[handle]:
        return 1
    else:
        return -1

def random_exploration():
    episode=[]
    num=np.zeros(n_arms)
    wins=np.zeros(n_arms)

    for i in range(n_trial):
        h=np.random.randint(0,n_arms)
        reward=pull_bandit(h)
        episode.append([h,reward])
        num[h]+= 1
        wins[h]+=1 if reward==1 else 0
    return episode, (num,wins )

e,r = random_exploration()

print("손잡이 형상 확률:", ["%.4f" % (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i
in range(n_arms)])

print("손잡이 추수($):", ["%.1d" % (2*r[1][i]-r[0][i]) for i in range(n_arms)])
print("순 수익($):", sum(np.asarray(e)[:,1]))


# e-탐욕을 구현하는 함수
def epsilon_greedy(eps):
    episode=[]
    num=np.zeros(n_arms)  # 손잡이별로 당긴 횟수
    wins=np.zeros(n_arms)  # 손잡이별로 승리 횟수

    for i in range(n_trial):
        r=np.random.random()
        if(r < eps or sum(wins) == 0):  # 확률 eps로 인한 선택
            h=np.random.randint(0, n_arms)
        else:
            prob=np.asarray([wins[i]/num[i] if num[i] > 0 else 0.0 for i in range(n_arms)])
            prob=prob/sum(prob)
            h=np.random.choice(range(n_arms), p=prob)

        reward=pull_bandit(h)
        episode.append([h, reward])
        num[h] += 1
        wins[h] = 1 if reward == 1 else 0
    return episode, (num, wins)

e, r=epsilon_greedy(0.1)
print("손잡이 형상 확률:", ["%.4f" % (r[1][i]/r[0][i]) if r[0][i] > 0 else 0.0 for i in range(n_arms)])
print("손잡이 추수($):", ["%.1d" % (2*r[1][i]-r[0][i]) for i in range(n_arms)])
print("순 수익($):", sum(np.asarray(e)[:,1]))
