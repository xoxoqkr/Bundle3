# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import operator
import itertools
from Class import Task
from Basic_Func import RouteTime, distance, FLT_Calculate, ActiveRiderCalculator, WillingtoWork
from Bundle_Func import BundleConsist
import numpy as np
import random
import copy
import time
import math


def CountActiveRider(riders, t, min_pr = 0, t_now = 0):
    """
    현대 시점에서 t 시점내에 주문을 선택할 확률이 min_pr보다 더 높은 라이더를 계산
    @param riders: RIDER CLASS DICT
    @param t: t시점
    @param min_pr: 최소 확률(주문 선택확률이 min_pr 보다는 높아야 함.)
    @return: 만족하는 라이더의 이름. LIST
    """
    names = []
    for rider_name in riders:
        rider = riders[rider_name]
        if ActiveRiderCalculator(rider, t_now, option = 'test') == True and rider.select_pr(t) >= min_pr:
            names.append(rider_name)
    return names


def WeightCalculator(riders, rider_names, sample_size = 1000):
    """
    시뮬레이션을 활용해서, t_i의 발생 확률을 계산함.
    @param riders: RIDER CLASS DICT
    @param rider_names: active rider names list
    @param sample_size: 포아송 분포 샘플 크기
    @return: w = {KY : ^_t_i ELE : ^_t_i가 발생할 확률}
    """
    w = {}
    ava_combinations = list(itertools.permutations(rider_names, len(rider_names))) # 가능한 조합
    count = {}
    for info in ava_combinations:
        #print(info)
        #input('확인')
        count[info] = 0
    poisson_dist = {}
    for rider_name in rider_names:
        rider = riders[rider_name]
        mu = rider.search_lamda
        if mu not in poisson_dist:
            poisson_dist[mu] = np.random.poisson(mu, sample_size)
            #poisson_ = poisson(mu)# 분포 추정
            #poisson_dist[mu] = poisson_.rvs(sample_size)  # 추정한 분포 기반 Sample 1000개 생성
    for _ in range(sample_size):
        tem = []
        for rider_name in rider_names:
            rider = riders[rider_name]
            val = random.choice(poisson_dist[rider.search_lamda])
            tem.append([rider_name, val])
        tem.sort(key = operator.itemgetter(1))
        seq = []
        for info in tem:
            seq.append(info[0])
        count[tuple(seq)] += 1
    for key in count:
        if count[key] > 0:
            w[key] = round(count[key]/sample_size,6)

    return w


def Calculate_e_b(stores, riders, infos, t_now):
    # todo : t시점이 흐르고 난 다음의 e_b가 계산되어야 함.
    e_b = []
    for store_name in stores:
        store = stores[store_name]
        for info in infos:
            rider = riders[info[0]]
            print('가게 {} 대기 중 고객 수 {}'.format(store.name, len(store.received_orders)))
            val = len(store.received_orders)*distance(rider.CurrentLoc(t_now), store.location)
            e_b.append(val)
    return round(sum(e_b),4)


def Calculate_e_b2(orders, stores, riders, infos, t_now):
    # todo : t시점이 흐르고 난 다음의 e_b가 계산되어야 함.
    e_b = []
    res = []
    for store_name in stores:
        res.append(0)
    for customer_name in orders:
        customer = orders[customer_name]
        if customer.time_info[4] == None:
            res[customer.store] += 1
    for info in infos:
        rider = riders[info[0]]
        for store_name in stores:
            store = stores[store_name]
            received_order_num = res[store_name]
            val = received_order_num * distance(rider.CurrentLoc(t_now), store.location)
            e_b.append(val)
            #print('가게 {} 대기 중 고객 수 {}'.format(store.name, len(store.received_orders)))
    return round(sum(e_b),4)

def Calculate_d_b(riders, states , t_now):
    # todo : t시점이 흐르고 난 다음의 d_b가 계산되어야 함.
    d_b = []
    for info in states:
        rider = riders[info[0]]
        val = WillingtoWork(rider, t_now)
        d_b.append(val)
    # state가 주어졌을 때의 d_b를 계산
    return round(sum(d_b),4)


def BundleScoreSimulator(riders, platform, orders, stores, w, t, t_now):
    # 매 번들 마다 수행해야함.
    e = []
    d = []
    # 라이더가 t_i의 순서대로 주문을 선택할 때, 선택할 주문을 선택
    for seq in w:
        tem = []
        for rider_name in seq:
            rider = riders[rider_name]
            current_loc = rider.CurrentLoc(t_now)
            #print('가짜', t_now)
            select_order_name = rider.OrderSelect(platform, orders, current_loc = current_loc)
            tem.append([rider_name, select_order_name])
        #e_b_bar = Calculate_e_b(stores, riders, tem, t_now)
        e_b_bar = Calculate_e_b2(orders, stores, riders, tem, t_now) #todo : 함수 수정
        d_b_bar = Calculate_d_b(riders, tem, t_now)
        e.append(w[seq]*e_b_bar)
        d.append(w[seq]*d_b_bar)
        #print('라이더 순서 {} W {} e {} d {}'.format(seq,w[seq],e_b_bar,d_b_bar))
        #라이더의 주문 선택 시뮬레이터
    return round(sum(e),4), round(sum(d),4)


def MIN_OD_pair(orders, q,s,):
    # 1 OD-pair 계산
    Q = itertools.permutations(q, s)  # 기존 OD pair의 가장 짧은 순서를 결정 하기 위함.
    OD_pair_dist = []
    for seq in Q:
        route_dist = 0
        tem_route = []
        for name in seq:
            tem_route += [orders[name].store_loc, orders[name].location]
        for index in range(1, len(tem_route)):
            before = tem_route[index - 1]
            after = tem_route[index]
            route_dist += distance(before, after)
        OD_pair_dist.append(route_dist)
    return min(OD_pair_dist)


def ParetoDominanceCount(datas, index, score_index1, score_index2, result_index, strict_option = False):
    """
    주어진 데이터에 대해 ParetoDominanceCount를 수행.
    @param datas: 대상이 되는 리스트
    @param index: 리스트 내의 이름-id- index
    @param score_index1: 점수 1
    @param score_index2: 점수 2
    @param result_index: datas에서 결과가 저장되는 index
    @param strict_option: pareto dominance count에서 dominance가 <=(False) or <(True) 인지
    @return: datas
    """
    count = 0
    for base_data in datas:
        dominance_count = 0
        for data in datas:
            #print(base_data, data)
            #input('확인')
            if base_data[index] != data[index]:
                if strict_option == True:
                    if base_data[score_index1] < data[score_index1] and base_data[score_index2] < data[score_index2]:
                        dominance_count += 1
                else:
                    if base_data[score_index1] <= data[score_index1] and base_data[score_index2] <= data[score_index2]:
                        dominance_count += 1
        datas[count][result_index] = dominance_count
        #input('확인')
        count += 1
    datas.sort(key = operator.itemgetter(result_index), reverse = True)
    return datas


def BundleConsideredCustomers(target_order, platform, riders, customers, speed = 1, bundle_search_variant = True, d_thres_option = True):
    not_served_ct_name_cls = {}
    not_served_ct_names = [] #번들 구성에 고려될 수 있는 고객들
    in_bundle_names = []
    for order_index in platform.platform:
        order = platform.platform[order_index]
        if order.type == 'bundle':
            in_bundle_names.append(order.customers)
    for customer_name in customers:
        customer = customers[customer_name]
        if customer.time_info[1] == None and customer.time_info[2] == None:
            if customer.type == 'single_order' and customer_name not in in_bundle_names:
                pass
            else:
                if bundle_search_variant == True:
                    pass
                else:
                    continue
            if d_thres_option == False:
                d_thres = 1000
            else:
                d_thres = customer.p2
            dist = distance(target_order.store_loc, customer.store_loc) / speed
            if target_order.name != customer.name and dist <= d_thres:
                not_served_ct_names.append(customer_name)
                not_served_ct_name_cls[customer_name] = customer
    current_in_bundle = []
    current_in_single = []
    for order_index in platform.platform:
        order = platform.platform[order_index]
        if order.type == 'bundle':
            current_in_bundle += platform.platform[order_index].customers
        else:
            current_in_single += platform.platform[order_index].customers
    rider_on_hand = []
    rider_finished = []
    for rider_name in riders:
        rider = riders[rider_name]
        rider_on_hand += rider.onhand
        rider_finished += rider.served
    res = {}
    for ct_name in not_served_ct_names:
        if ct_name in rider_on_hand + rider_finished:
            input('ERROR {} :: 고려 고객 {} 제외1 {} 제외 2 {}'.format(ct_name, not_served_ct_names, rider_on_hand, rider_finished))
        else:
            res[ct_name] = customers[ct_name]
    res[target_order.name] = target_order
    return res


def SelectByTwo_sided_way2(target_order, riders, orders, stores, platform, p2, t, t_now, min_pr, thres = 0.1, speed = 1, bundle_permutation_option = False, unserved_bundle_order_break = True, s = 3, scoring_type = 'myopic',input_data = None, input_weight = None):
    """
    주어진 feasible bundle(혹은 target order를 기준으로 탐색된 feasible bundle)에
    대해서 s,e,d 점수가 높은 번들을 선택 후 제안.
    @param target_order: 탐색의 기준이 되는 주문
    @param riders: RIDER CLASS DICT
    @param orders: CUSTOMER CLASS DICT
    @param stores: STORE CLASS DICT
    @param platform: PLATFORM CLASS DICT
    @param p2: max FLT(고객 시간 제한)
    @param t:다음 주문이 발생할 것으로 예상되는 시간 간격
    @param t_now: 현재 시간
    @param min_pr: 라이더 주문 탐색 확률 최솟 값(이 값보다 확률이 작은 경우는 계산에 고려X)
    @param speed: 차량 속도
    @param bundle_search_variant: 번들 탐색시 대상이 되는 고객들 결정 (True : 기존에 번들의 고객들은 고려 X , False : 기존 번들의 고객도  고려)
    @param s: 고려되는 번들 크기 default = 3
    @param input_data: feasible_bundles을 외부에서 계산하는 경우에 데이터 입력
    @return:
    """
    if input_data == None:
        print('대상 고객 이름 {} 대상 수 {}'.format(target_order.name, len(orders)))
        B3 = ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, speed=speed, bundle_permutation_option = bundle_permutation_option)
        B2 = ConstructFeasibleBundle_TwoSided(target_order, orders, s - 1, p2, speed=speed,bundle_permutation_option=bundle_permutation_option)
        feasible_bundles = B2 + B3
        if len(feasible_bundles) > 0:
            comparable_b = []
            feasible_bundles.sort(key=operator.itemgetter(6))  # s_b 순으로 정렬  #target order를 포함하는 모든 번들에 대해서 s_b를 계산.
            b_star = feasible_bundles[0][6]
            for ele in feasible_bundles:
                if (ele[6] - b_star) / b_star <= thres:  # percent loss 가 thres 보다 작아야 함.
                    comparable_b.append(ele)
            feasible_bundles = comparable_b
        print('input_data == None :: ## {}'.format(len(feasible_bundles)))
    else:
        feasible_bundles = input_data
        print('input_data != None')
    count = 0
    scores = []
    print('대상 번들들 {}'.format(feasible_bundles))
    if input_weight == None:
        active_rider_names = CountActiveRider(riders, t, min_pr=min_pr, t_now=t_now)
        weight = WeightCalculator(riders, active_rider_names)
        w_list = list(weight.values())
        try:
            input('T {} / 대상 라이더 수 {}/시나리오 수 {} 중 {} / w평균 {} /w표준편차 {}'.format(t_now,len(active_rider_names),math.factorial(len(active_rider_names)),len(weight), np.average(w_list), np.std(w_list)))
        except:
            input('T {} 출력 에러'.format(t_now))
    else:
        weight = input_weight
    for feasible_bundle in feasible_bundles:
        s = feasible_bundle[6]
        e_pool = []
        d_pool = []
        e = 0
        d = 0
        if scoring_type == 'two_sided':
            try:
                e,d = Two_sidedScore(feasible_bundle, riders, orders, stores, platform, t, t_now, min_pr, M=1000, sample_size=1000, weight = weight)
                e_pool.append(e)
                d_pool.append(d)
            except:
                e = 1000000
                d = 1000000
                pass
        scores.append([count, s,e,d,0])
        count += 1
    scores.sort(key = operator.itemgetter(1), reverse = True)
    if scoring_type == 'myopic':
        sorted_scores = scores
    else:
        sorted_scores = ParetoDominanceCount(scores, 0, 2, 3, 4, strict_option = False)
        print('scored datas :: {}'.format(sorted_scores))
    if len(feasible_bundles) > 0:
        res = feasible_bundles[sorted_scores[0][0]] + sorted_scores[0][1:4] + [0]
        return res
    else:
        return None


def ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, thres = 0.05, speed = 1, bundle_permutation_option = False, uncertainty = False, platform_exp_error = 1):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param new_orders: new order genrated during t_bar
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param p2: max FLT
    :param speed:rider speed
    :parm option:
    :parm uncertainty:
    :parm platform_exp_error:
    :parm bundle_search_variant: 번들 탐색시 대상이 되는 고객들 결정 (True : 기존에 번들의 고객들은 고려 X , False : 기존 번들의 고객도  고려)
    :return: constructed bundle set
    """
    d = []
    for customer_name in orders:
        if customer_name != target_order.name:
            d.append(customer_name)
    print('대상 고객 {} 고려 고객들 {} '.format(target_order.name, d))
    if len(d) > s - 1:
        M = itertools.permutations(d, s - 1)
        b = []
        for m in M:
            #print(list(m))
            q = list(m) + [target_order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            #print('확인 1 {} : 확인2 {}'.format(subset_orders, time_thres))
            tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, bundle_permutation_option= bundle_permutation_option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, feasible_return = True)
            if len(tem_route_info) > 0:
                OD_pair_dist = MIN_OD_pair(orders, q, s)
                for info in tem_route_info:
                    info.append((OD_pair_dist - info[5] / s))
            b += tem_route_info
        #print('경우의 수 {} 가능 번들 수 {} : 정보 d {} s {}'.format(len(list(M)), len(b), d, s))
        comparable_b = []
        if len(b) > 0:
            b.sort(key=operator.itemgetter(6))  # s_b 순으로 정렬  #target order를 포함하는 모든 번들에 대해서 s_b를 계산.
            b_star = b[0][6]
            for ele in b:
                if (ele[6] - b_star)/b_star <= thres: #percent loss 가 thres 보다 작아야 함.
                    comparable_b.append(ele)
        return comparable_b
    else:
        return []

def Two_sidedScore(bundle, riders, orders, stores, platform, t, t_now, min_pr , M = 1000, sample_size=1000, platform_exp_error = 1, weight = None):
    if weight == None:
        active_rider_names = CountActiveRider(riders, t, min_pr=min_pr, t_now = t_now)
        p_s_t = WeightCalculator(riders, active_rider_names, sample_size=sample_size)
    else:
        p_s_t = weight
    w_list = []
    for p in p_s_t:
        #w_list.append(p[1])
        w_list.append(p_s_t[p])
    print('시나리오 수 {} / w평균 {} /w표준편차 {}'.format(len(p_s_t), np.average(w_list), np.std(w_list)))
    #print('T: {} 길이 {} 평균 {} 분산 {}'.format(t_now, len(p_s_t), np.average(w_list), np.var(w_list)))
    #input('w 확인')
    mock_platform = copy.deepcopy(platform)
    mock_index = max(mock_platform.platform.keys()) + 1
    route = []
    for node in bundle[0]:
        if node >= M:
            customer_name = node - M
            customer = orders[customer_name]
            route.append([customer_name, 0, customer.store_loc, 0])
        else:
            customer_name = node
            customer = orders[customer_name]
            route.append([customer_name, 1, customer.location, 0])
    fee = 0
    for customer_name in bundle[4]:
        fee += orders[customer_name].fee  # 주문의 금액 더하기.
        orders[customer_name].in_bundle_time = t_now
        pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
        orders[customer_name].platform_exp_cook_time = random.choice(pool)
    o = Task(mock_index, bundle[4], route, 'bundle', fee=fee)
    o.average_ftd = bundle[2]
    mock_platform.platform[mock_index] = o #가상의 번들을 추가.
    e,d = BundleScoreSimulator(riders, mock_platform, orders, stores, p_s_t, t, t_now)
    return e,d



