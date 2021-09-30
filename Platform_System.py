# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import time
from Bundle_Func import CountUnpickedOrders, CalculateRho, RequiredBreakBundleNum, BreakBundle, PlatformOrderRevise4, LamdaMuCalculate, NewCustomer
from Platform_Func import ParetoDominanceCount, BundleConsideredCustomers, SelectByTwo_sided_way2, Two_sidedScore, CountActiveRider, WeightCalculator
import operator
from Bundle_selection_problem import Bundle_selection_problem2
import math
import numpy as np


def Platform_process4(env, platform_set, orders, riders, stores, p2,thres_p,interval, bundle_permutation_option = False, speed = 1, end_t = 1000, min_pr = 0.05, divide_option = False,\
                      unserved_bundle_order_break = True,  scoring_type = 'myopic',bundle_selection_type = 'greedy', considered_customer_type = 'new'):
    while env.now <= end_t:
        now_t = env.now
        lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, now_t, interval=interval, return_type='class')
        p = CalculateRho(lamda1, lamda2, mu1, mu2)
        if p >= thres_p:
            B = []
            if considered_customer_type == 'new':
                considered_customers_names = NewCustomer(orders, now_t, interval = interval)
            else:
                considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval = interval,  return_type='name')
            print('탐색 대상 고객들 {}'.format(considered_customers_names))
            active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t)
            weight2 = WeightCalculator(riders, active_rider_names) #todo: WeightCalculator가 신뢰할 만한 값을 주고 있는가?
            w_list = list(weight2.values())
            try:
                input('T {} / 대상 라이더 수 {}/시나리오 수 {} 중 {} / w평균 {} /w표준편차 {}'.format(now_t, len(active_rider_names),math.factorial(len(active_rider_names)),len(weight2), np.average(w_list),np.std(w_list)))
            except:
                input('T {} 출력 에러'.format(now_t))
            for customer_name in considered_customers_names:
                start = time.time()
                target_order = orders[customer_name]
                considered_customers = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                                 bundle_search_variant=unserved_bundle_order_break,
                                                                 d_thres_option=True, speed=speed)
                selected_bundle = SelectByTwo_sided_way2(target_order, riders, considered_customers, stores, platform_set, p2, interval, env.now, min_pr, speed=speed, \
                                                         scoring_type = scoring_type,bundle_permutation_option= bundle_permutation_option,\
                                                         unserved_bundle_order_break=unserved_bundle_order_break, input_weight= weight2)
                end = time.time()
                print('고객 당 계산 시간 {} : 선택 번들1 {}'.format(end - start, selected_bundle))
                # selected_bundle 구조 : [(1151, 1103, 103, 151), 16.36, 10.69, 5.03, [103, 151], 16.36, 23.1417(s), 23.1417(s), 1000000(e), 1000000(d), 0]
                if selected_bundle != None:
                    B.append(selected_bundle)
            #Part2 기존에 제시되어 있던 번들 중 새롭게 구성된 번들과 겹치는 부분이 있으면 삭제해야 함.
            if unserved_bundle_order_break == False:
                pass
            else:
                for order_index in platform_set.platform:
                    order = platform_set.platform[order_index]
                    if order.type == 'bundle':
                        print('확인 {}'.format(order.old_info))
                        order.old_info += [order.old_info[6]]
                        e,d = Two_sidedScore(order.old_info, riders, orders, stores, platform_set , interval, now_t, min_pr, M=1000,
                                       sample_size=1000, platform_exp_error=1, weight = weight2)
                        order.old_info += [e,d,0]
                        B.append(order.old_info)
            unique_bundles = [] #P의 역할
            if len(B) > 0:
                if scoring_type == 'myopic':
                    print('정렬 정보{}'.format(B))
                    B.sort(key = operator.itemgetter(6))
                else:
                    B = ParetoDominanceCount(B, 0, 8, 9, 10, strict_option = False)
                #Part 2 -1 Greedy한 방식으로 선택
                selected_customer_name_check = [] #P의 확인용
                #unique_bundles = [] #P의 역할
                if bundle_selection_type == 'greedy':
                    for bundle_info in B:
                        duplicate = False
                        for ct_name in bundle_info[4]:
                            if ct_name in selected_customer_name_check:
                                duplicate = True
                                break
                        if duplicate == True:
                            continue
                        else:
                            unique_bundles.append(bundle_info[:7])
                            selected_customer_name_check += bundle_info[4]
                else: # set cover problem 풀이
                    feasiblity, unique_bundles = Bundle_selection_problem2(B)
                    print('결과확인 {} : {}'.format(feasiblity, unique_bundles))
            #part 3 Upload P
            new_orders = PlatformOrderRevise4(unique_bundles, orders, platform_set, now_t = now_t, unserved_bundle_order_break = unserved_bundle_order_break, divide_option = divide_option)
            platform_set.platform = new_orders
        else:
            org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres=thres_p)
            if sum(rev_bundle_num) < sum(org_bundle_num):
                break_info = [org_bundle_num[0] - rev_bundle_num[0],org_bundle_num[1] - rev_bundle_num[1]] #[B2 해체 수, B3 해체 수]
                platform_set.platform = BreakBundle(break_info, platform_set, orders)
        yield env.timeout(interval)