import numpy as np
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
from copy import deepcopy
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
from scipy.fftpack import fftshift,fftn
from matplotlib.pyplot import MultipleLocator
depthmap1 = np.load('./data_q1.npy')    #使用numpy载入npy文件b
#depthmap1 = depthmap1[20,:,:]
print(depthmap1.shape)
gamma = 78.986*10**12
Ts = 1.25*10**(-7)
T = 3.2*10**(-5)
f0 = 78.8*10**9
light_speed = 3*10**8
#带宽
B = gamma*T
Nf = 32
L = 0.0815
Na = 86
print('带宽',B)
print('最小距离',light_speed/2/B)
#print('最大距离',T*shinec/2)
def A_tconj(A):
    A = np.array(A)
    B = deepcopy(A.T)
    (rows,cols) = A.shape
    for i in range(rows):
        for j in range(cols):
            B[j,i] = A[i,j].conjugate()
    return B
def DFT(N,X):
    '''
    #N :长度
    #X : 列表
    '''
    list1=[]
    for i in range(N):
        sumx=0
        for k in range(N):
            sumx+=X[k]*pow(np.e,-(2*np.pi*i*k/N)*1j)
        sumx=sumx/N
        #sumx=abs(sumx)
        list1.append(sumx)
    return list1
def get_distance(n,N):
    #给距离
    gamma = 78.986*10**12
    light_speed = 3*10**8
    Ts = 1.25*10**(-7)
    res = n/N*light_speed/gamma/Ts/2
    return res
def get_angle(n,N):
    #给角度
    if n == N/2:
        return np.pi/2
    f0 = 78.8*10**9
    light_speed = 3*10**8
    L = 0.0815
    if n>N/2:
        res = np.arcsin( (n-N)/N*light_speed*85/f0/L )
    else:
        res = np.arcsin( (n/N)*light_speed*85/f0/L )
    return res
def OMP_K(Y256,K = 2,accuracy_N = 1024,original_N = 256,index_probably = []):
    #给定K值，强行找出K个值
    #Y256 = np.array([depthmap1[0]]).T
    #K = 2
    #accuracy_N = 1024  #512  #要提高的精度
    #original_N = 256
    #index_probably = list(range(450,500)) #list(range(1800,2000))  #搜索范围，也是K的最大值  ()
    #index_probably = index_probably
    Y256 = np.array([Y256]).T
    w_exp = np.exp(1j*2*np.pi/accuracy_N)
    index_bag = []
    not_index_bag = list(set(index_probably)-set(index_bag))
    rkold = Y256
    x_yuanshi_old = 0+0j
    y_moold = np.sqrt(sum([abs(kj[0])**2 for kj in Y256]))/256/2
    for ytu in range(len(index_probably)):


        #最多循环k次，就不再搜索了
        #print('---')

        faiirk = [] #一列数据都算一下最大值
        for index_argmax in not_index_bag:

            Yhat_fairk = 0.0 + 0.0j #一个相乘数据并求和的记录
            for zn_256 in range(original_N):
                Yhat_fairk += (w_exp**(zn_256*index_argmax)).conjugate()*rkold[zn_256,0]
            faiirk.append(abs(Yhat_fairk))
        #print(faiirk)
        #print(faiirk)
        index_find = faiirk.index(max(faiirk))  #这里找的是k
        #表格变换
        value_index_find = not_index_bag[index_find]
        index_bag.append(value_index_find)
        not_index_bag = list(set(index_probably)-set(index_bag))
        index_bag.sort()
        #matrix_fai = np.zeros((original_N,len(index_bag)))
        matrix_fai = []

        for row in range(original_N):
            row_bag = []
            for col in range(len(index_bag)):
                row_bag.append(w_exp**(col*row))
            matrix_fai.append(row_bag)    
        matrix_fai = np.array(matrix_fai)
        if len(index_bag)<=1:
            rk_guji =Y256 -  np.linalg.inv(np.dot(A_tconj(matrix_fai),matrix_fai))[0,0] * np.dot( np.dot(matrix_fai,A_tconj(matrix_fai)),Y256)
        else:
            rk_guji = Y256 - np.dot(np.dot(np.dot(matrix_fai,np.linalg.inv(np.dot(A_tconj(matrix_fai),matrix_fai))),A_tconj(matrix_fai)),Y256)

        x_kpanduanpart2_matrix_fai = []
        for row in range(original_N):
            row_bag = []
            for col in range(1):
                row_bag.append(w_exp**(not_index_bag[index_find]*row))
            x_kpanduanpart2_matrix_fai.append(row_bag)    
        x_kpanduanpart2_matrix_fai = np.array(x_kpanduanpart2_matrix_fai)
        x_yuanshi_new = np.linalg.inv(np.dot(A_tconj(x_kpanduanpart2_matrix_fai),x_kpanduanpart2_matrix_fai))[0,0]*(np.dot(A_tconj(x_kpanduanpart2_matrix_fai),Y256)[0,0])
        delta_x = abs(x_yuanshi_new - x_yuanshi_old)
        if len(index_bag) == K:#delta_x<= y_moold:
            #输出 X
            #index_bag.pop(index_bag.index(value_index_find))
            index_bag.sort()
            #print(index_bag)
            part2_matrix_fai = []
            for row in range(original_N):
                row_bag = []
                for col in range(len(index_bag)):
                    row_bag.append(w_exp**(index_bag[col]*row))
                part2_matrix_fai.append(row_bag)    
            part2_matrix_fai = np.array(part2_matrix_fai)
            #print(part2_matrix_fai[:,0:3])
            if len(index_bag)<=1:
                x_index_out = (np.linalg.inv(np.dot(A_tconj(part2_matrix_fai),part2_matrix_fai))[0,0])*(np.dot(A_tconj(part2_matrix_fai),Y256)[0,0])
            else:
                x_index_out = np.dot((np.linalg.inv(np.dot(A_tconj(part2_matrix_fai),part2_matrix_fai))),(np.dot(A_tconj(part2_matrix_fai),Y256)))
            break
        else:
            x_yuanshi_old = deepcopy(x_yuanshi_new)
            rkold = deepcopy(rk_guji)
    return index_bag,x_index_out

import pickle

#保存和读取数据
def pickle_save(filename,v,describe):#变量名称地址，变量名，描述  return ok
    f=open(filename,'wb')
    pickle.dump([v,describe],f)
    f.close()
    return 'ok!'
def pickle_read(filename):#变量名称地址, return 变量，描述
    f=open(filename,'rb')
    [r,describe]=pickle.load(f)
    f.close()
    return r,describe
def intermediate_frequency_signal_K(distance_K,degree_angle_K,K):
    '''
    根据找到的距离和角度得到对应的模拟信号
    return 86*256
    '''
    rows = 86
    cols = 256
    K = K
    gamma = 78.986*10**12
    Ts = 1.25*10**(-7)
    light_speed = 3*10**8
    simulation_signal = []#np.zeros((rows,cols))
    
    for row in range(rows):
        simulation_signal_k1 = np.array([0j]*cols)
        xn = -0.0815/2 + row * 0.0815/85
        yn = 0.0
        for k in range(K):
            Xk = distance_K[k]*np.sin(degree_angle_K[k])
            Yk = distance_K[k]*np.cos(degree_angle_K[k])
            Rnk = 2*np.sqrt( (xn-Xk)**2 + (yn-Yk)**2 )
            for col in range(cols):
                snkt = 1 * np.exp( 1j*( 2*np.pi*gamma*Ts*col*Rnk/light_speed + 2*np.pi*f0*Rnk/light_speed ) )
                simulation_signal_k1[col] += snkt
        simulation_signal.append(simulation_signal_k1)
    simulation_signal = np.array(simulation_signal)
    return simulation_signal
from scipy.optimize import differential_evolution
def myfun(v):
    jvzheng = depthmap1
    l,jiao = v
    return matrix_loss(intermediate_frequency_signal_K([l],[jiao],1),jvzheng)
def myfun_2(v):
    jvzheng = depthmap1
    l,jiao,l2,jiao2 = v
    return matrix_loss(intermediate_frequency_signal_K([l,l2],[jiao,jiao2],2),jvzheng)
r_min, r_max = -5.0, 5.0
jvzheng = depthmap1
# define the bounds on the search
bounds = [[6.05328-2*0.05934, 6.05328+2*0.05934], [-0.03, 0.03],[6.05328-2*0.05934, 6.05328+2*0.05934], [-0.03, 0.03]]
result = differential_evolution(myfun_2, bounds)

zuobiao = []
for lenth in range(2,32):
    print(lenth)
    zuobiao1 = []
    jiaodutu = []
    for jiao in range(86):
        Y256 = depthmap1[lenth,jiao,:]
        out_dis = DFT(256,Y256)#OMP(Y256,threshold_rk = 2,accuracy_N = 256*2,original_N = 256,index_probably = list(range(256*2)))
        if jiao ==0:
            out_dis_abs = [abs(aa) for aa in out_dis]
            jishu = []
            for i in range(256):
                if out_dis_abs[i]> max(out_dis_abs)*0.5 and out_dis_abs[i]>= max(out_dis_abs[max(0,i-1):min(255,i+2)]):
                    jishu.append(i)
            #if lenth == 4:
                #plt.plot(out_dis_abs)
            zuobiao1.append([jishu[0],jishu[1]])
        jiaodutu.append([out_dis[jishu[0]],out_dis[jishu[1]]])
    #print(jiaodutu)
    #break
    #jiaodutu = np.array(jiaodutu) np.array(jiaodutu)
    #print(jiaodutu.shape)
    #if lenth>=2:

    jiaodutu = np.array(jiaodutu).T
    #out_dis1 = OMP_K(jiaodutu[0],K = 1,accuracy_N = 86*2,original_N = 86,index_probably = list(range(86*2)))
    out_dis = DFT(86,jiaodutu[0])#OMP(Y256,threshold_rk = 2,accuracy_N = 256*2,original_N = 256,index_probably = list(range(256*2)))
    out_dis_abs = [abs(aa) for aa in out_dis]
    an1 = out_dis_abs.index(max(out_dis_abs))
    #out_dis2 = OMP_K(jiaodutu[1],K = 1,accuracy_N = 86*2,original_N = 86,index_probably = list(range(86*2)))
    out_dis = DFT(86,jiaodutu[1])#OMP(Y256,threshold_rk = 2,accuracy_N = 256*2,original_N = 256,index_probably = list(range(256*2)))
    out_dis_abs = [abs(aa) for aa in out_dis]
    an2 = out_dis_abs.index(max(out_dis_abs))
    zuobiao1.append([an1,an2])
    zuobiao.append(zuobiao1)
    
def OMP(Y256,threshold_rk = 2,accuracy_N = 1024,original_N = 256,index_probably = []):
    #Y256 = np.array([depthmap1[0]]).T
    #threshold_rk = 2
    #accuracy_N = 1024  #512  #要提高的精度
    #original_N = 256
    #index_probably = list(range(450,550))  #搜索范围，也是K的最大值  ()
    index_probably = index_probably
    Y256 = np.array([Y256]).T
    w_exp = np.exp(1j*2*np.pi/accuracy_N)
    index_bag = []
    not_index_bag = list(set(index_probably)-set(index_bag))
    rkold = Y256
    for ytu in range(len(index_probably)):
    #最多循环k次，就不再搜索了
        #print('---')
        faiirk = [] #一列数据都算一下最大值
        for index_argmax in not_index_bag:

            Yhat_fairk = 0.0 + 0.0j #一个相乘数据并求和的记录
            for zn_256 in range(original_N):
                Yhat_fairk += (w_exp**(zn_256*index_argmax)).conjugate()*rkold[zn_256,0]
            faiirk.append(abs(Yhat_fairk))
        #print(faiirk)
        #print(faiirk)
        index_find = faiirk.index(max(faiirk))  #这里找的是k
        #表格变换
        index_bag.append(not_index_bag[index_find])
        not_index_bag = list(set(index_probably)-set(index_bag))
        index_bag.sort()
        #matrix_fai = np.zeros((original_N,len(index_bag)))
        matrix_fai = []

        for row in range(original_N):
            row_bag = []
            for col in range(len(index_bag)):
                row_bag.append(w_exp**(col*row))
            matrix_fai.append(row_bag)    
        matrix_fai = np.array(matrix_fai)
        if len(index_bag)<=1:
            rk_guji =Y256 -  np.linalg.inv(np.dot(A_tconj(matrix_fai),matrix_fai))[0,0] * np.dot( np.dot(matrix_fai,A_tconj(matrix_fai)),Y256)
        else:
            rk_guji = Y256 - np.dot(np.dot(np.dot(matrix_fai,np.linalg.inv(np.dot(A_tconj(matrix_fai),matrix_fai))),A_tconj(matrix_fai)),Y256)
        fanshu_r_left = [jj.real**2+jj.imag**2 for jj in (rk_guji-rkold).T[0] ]
        r_bijiao_left = np.sqrt(sum(fanshu_r_left))
        fanshu_r_right = [jj.real**2+jj.imag**2 for jj in (rk_guji).T[0] ]
        r_bijiao_right = np.sqrt(sum(fanshu_r_right))
        if r_bijiao_left <= r_bijiao_right*threshold_rk:
            break
            #print(index_find)
            rkold = deepcopy(rk_guji)
        else:
            rkold = deepcopy(rk_guji)
        #print(index_bag)
    #print(index_bag)
    index_bag.sort()
    #print(index_bag)
    part2_matrix_fai = []
    for row in range(original_N):
        row_bag = []
        for col in range(len(index_bag)):
            row_bag.append(w_exp**(index_bag[col]*row))
        part2_matrix_fai.append(row_bag)    
    part2_matrix_fai = np.array(part2_matrix_fai)
    #print(part2_matrix_fai[:,0:3])
    if len(index_bag)<=1:
        x_index_out = (np.linalg.inv(np.dot(A_tconj(part2_matrix_fai),part2_matrix_fai))[0,0])*(np.dot(A_tconj(part2_matrix_fai),Y256)[0,0])
    else:
        x_index_out = np.dot((np.linalg.inv(np.dot(A_tconj(part2_matrix_fai),part2_matrix_fai))),(np.dot(A_tconj(part2_matrix_fai),Y256)))
    return index_bag,x_index_out