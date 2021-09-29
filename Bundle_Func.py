# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import operator
import itertools
from Basic_Func import RouteTime, distance, FLT_Calculate, ActiveRiderCalculator, WillingtoWork
from Class import Task
import numpy as np
import random

def NewCustomer(cusotmers, now_t, interval = 5):
    new_customer_names = []
    for customer_name in cusotmers:
        customer = cusotmers[customer_name]
        if now_t - interval <= customer.time_info[0] and customer.time_info[1] == None:
            new_customer_names.append(customer.name)
    return new_customer_names


def LamdaMuCalculate(orders, riders, now_t, interval = 5, return_type = 'class'):
    unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval=interval, return_type=return_type)  # lamda1
    lamda1 = len(unpicked_orders)
    idle_riders, mu2 = CountIdleRiders(riders, now_t, interval=interval, return_type=return_type)
    mu1 = len(idle_riders)
    return lamda1, lamda2, mu1, mu2


def CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = 0, add_mu = 0):
    """
    Calculate rho
    :param lamda1: current lamda
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param add_lamda: additional lamda
    :param add_mu: additional mu
    :return: rho
    """
    if mu1 + mu2 + add_mu > 0:
        rho = (lamda1 + lamda2 + add_lamda) / (mu1 + mu2 + add_mu)
    else:
        rho = 2
    return round(rho, 4)


def RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres = 1):
    """
    Cacluate required b2 and b3 number
    condition : rho <= thres
    :param lamda1: current un-selected order
    :param lamda2: future generated order
    :param mu1: current rider
    :param mu2: future rider
    :param thres: rho thres: system over-load
    :return: b2, b3
    """
    b2 = 0
    b3 = 0
    for index in range(lamda1+lamda2):
        b2 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = -b2)
        #rho = (lamda1 + lamda2 - b2)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    for index in range(lamda1+lamda2):
        b2 -= 1
        b3 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda=-(b2+b3))
        #rho = (lamda1 + lamda2 - b2 - b3)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    return b2, b3


def RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres = 1):
    """
    Caclculate availiable break-down bundle number
    :param platform_set: orders set : [order,...]
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param thres: system level.
    :return:
    """
    org_b2_num = 0
    org_b3_num = 0
    b2_num = 0
    b3_num = 0
    customer_num = 0
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
                b2_num += 1
                org_b2_num += 1
            else:
                b3_num += 1
                org_b3_num += 1
        else:
            customer_num += 1
    end_para = False
    for count in range(org_b3_num): #break b3 first
        if b3_num > 0:
            b3_num -= 1
            customer_num += 3
        else:
            pass
        p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
        if p >= thres:
            end_para = True
            break
    if end_para == False: #if p < thres, than break b2
        for count in range(org_b2_num):
            if b2_num > 0:
                b2_num -= 1
                customer_num += 2
            else:
                pass
            p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
            if p >= thres:
                break
    return [org_b2_num,org_b3_num],[b2_num, b3_num]


def BreakBundle(break_info, platform_set, customer_set):
    """
    Break bundle by break_info
    And return the revised platform_set
    :param break_info: bundle breaking info [b2 decrcase num, b2 decrcase num]
    :param platform_set: orders set : [order,...]
    :param customer_set: customer set : [customer class,...]
    :return: breaked platform set
    """
    b2 = []
    b3 = []
    single_orders = []
    breaked_customer_names = []
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
                b2.append(order)
            else:
                b3.append(order)
        else:
            single_orders.append(order)
    b2.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    b3.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    for break_b2 in range(min(break_info[0],len(b2))):
        #breaked_customer_names.append(b2[0].customers)
        breaked_customer_names += b2[0].customers
        del b2[0]
    for break_b3 in range(min(break_info[1],len(b3))):
        #breaked_customer_names.append(b3[0].customers)
        breaked_customer_names += b3[0].customers
        del b3[0]
    breaked_customers = []
    order_nums = []
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        order_nums += order.customers
    order_num = max(order_nums) + 1
    for customer_name in breaked_customer_names:
        route = [[customer_name, 0, customer_set[customer_name].store_loc, 0],[customer_name, 1, customer_set[customer_name].location, 0 ]]
        order = Task(order_num,[customer_name], route, 'single', fee = customer_set[customer_name].fee)
        breaked_customers.append(order)
    res = {}
    for order in single_orders + b2 + b3 + breaked_customers:
        res[order.index] = order
    return res


def BundleConsist(orders, customers, p2, time_thres = 0, speed = 1,M = 1000, bundle_permutation_option = False, uncertainty = False, platform_exp_error =  1, feasible_return = False):
    """
    Construct bundle consists of orders
    :param orders: customer order in the route. type: customer class
    :param customers: customer dict  {[KY]customer name: [Value]class customer,...}
    :param p2: allowable FLT increase
    :param M: big number for distinguish order name and store name
    :param speed: rider speed
    :return: feasible route
    """
    order_names = [] #가게 이름?
    for order in orders:
        order_names.append(order.name)
    store_names = []
    for name in order_names:
        store_names.append(name + M)
    candi = order_names + store_names
    if bundle_permutation_option == False:
        subset = itertools.permutations(candi, len(candi))
    else:
        store_subset = itertools.permutations(store_names, len(store_names))
        store_subset = list(store_subset)
        order_subset = itertools.permutations(order_names, len(order_names))
        order_subset = list(order_subset)
        test = []
        test_names = itertools.permutations(order_names, 2)
        for names in test_names:
            dist = distance(customers[names[0]].location, customers[names[1]].location)
            if dist > 15:
                return []
        subset = []
        for store in store_subset:
            for order in order_subset:
                tem = store + order
                subset.append(tem)
        pass
    #print('번들 처리 시작. 대상 subset{}'.format(subset))
    #print('번들 고려시 탐색 수 {}'.format(len(list(subset))))
    feasible_subset = []
    for route in subset:
        #print('고객이름',order_names,'가게이름',store_names,'경로',route)
        sequence_feasiblity = True #모든 가게가 고객 보다 앞에 오는 경우.
        feasible_routes = []
        for order_name in order_names: # order_name + M : store name ;
            if route.index(order_name + M) < route.index(order_name):
                pass
            else:
                sequence_feasiblity = False
                break
        if sequence_feasiblity == True:
            ftd_feasiblity, ftds = FLT_Calculate(orders, customers, route, p2, [],M = M ,speed = speed, uncertainty =uncertainty, exp_error=platform_exp_error)
            #customer_in_order, customers, route, p2, except_names, M = 1000, speed = 1, now_t = 0
            if ftd_feasiblity == True:
                route_time = RouteTime(orders, route, speed=speed, M=M, uncertainty = uncertainty, error = platform_exp_error)
                feasible_routes.append([route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names,round(route_time, 2)])
                #print('시간 정보 번들 경로 시간 {} : 가능한 짧은 시간 {}'.format(route_time, time_thres))
                #if route_time < time_thres :
                #    feasible_routes.append([route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2)])
                #    input('번들 생성 절약 시간 {}'.format(time_thres - route_time))
                #[경로, 최대FTD, 평균FTD, 최소FTD]
        if len(feasible_routes) > 0:
            feasible_routes.sort(key = operator.itemgetter(2))
            feasible_subset.append(feasible_routes[0])
    if len(feasible_subset) > 0:
        feasible_subset.sort(key = operator.itemgetter(2))
        #GraphDraw(feasible_subset[0], customers)
        if feasible_return == True:
            return feasible_subset
        else:
            return feasible_subset[0]
    else:
        return []


def ConstructBundle(orders, s, n, p2, speed = 1, option = False, uncertainty = False, platform_exp_error = 1):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param n: needed bundle number
    :param p2: max FLT
    :param speed:rider speed
    :return: constructed bundle set
    """
    B = []
    for order_name in orders:
        order = orders[order_name]
        d = []
        dist_thres = order.p2
        for order2_name in orders:
            order2 = orders[order2_name]
            dist = distance(order.store_loc, order2.store_loc)/speed
            if order2 != order and dist <= dist_thres:
                d.append(order2.name)
        M = itertools.permutations(d, s - 1)
        #print('번들 구성 고려 subset 수 {}'.format(len(list(M))))
        #M = list(M)
        b = []
        for m in M:
            q = list(m) + [order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, bundle_permutation_option= option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error)
            if len(tem_route_info) > 0:
                b.append(tem_route_info)
        if len(b) > 0:
            b.sort(key = operator.itemgetter(2))
            B.append(b[0])
            #input('삽입되는 {}'.format(b[0]))
    #n개의 번들 선택
    B.sort(key = operator.itemgetter(5))
    selected_bundles = []
    selected_orders = []
    print('번들들 {}'.format(B))
    for bundle_info in B:
        # bundle_info = [[route,max(ftds),average(ftds), min(ftds), names],...,]
        unique = True
        for name in bundle_info[4]:
            if name in selected_orders:
                unique = False
                break
        if unique == True:
            selected_orders += bundle_info[4]
            selected_bundles.append(bundle_info)
            if len(selected_bundles) >= n:
                break
    if len(selected_bundles) > 0:
        #print("selected bundle#", len(selected_bundles))
        print("selected bundle#", selected_bundles)
        #input('멈춤7')
        pass
    #todo: 1)겹치는 고객을 가지는 번들 중 1개를 선택해야함. 2)어떤 번들이 더 좋은 번들인가?
    return selected_bundles


def CountUnpickedOrders(orders, now_t , interval = 10, return_type = 'class'):
    """
    return un-picked order
    :param orders: order list : [order class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: unpicked_orders, lamda2(future generated order)
    """
    unpicked_orders = []
    interval_orders = []
    for order_name in orders:
        order = orders[order_name]
        if order.time_info[1] == None:
            if return_type == 'class':
                unpicked_orders.append(order)
            elif return_type == 'name':
                unpicked_orders.append(order.name)
            else:
                pass
        if now_t- interval <= order.time_info[0] < now_t:
            interval_orders.append(order.name)
    return unpicked_orders, len(interval_orders)


def PlatformOrderRevise(bundle_infos, customer_set, order_index, platform_set, M = 1000, divide_option = False, now_t = 0, platform_exp_error = 1, new_type = False):
    """
    Construct unpicked_orders with bundled customer
    :param bundles: constructed bundles
    :param customer_set: customer list : [customer class,...,]
    :return: unserved customer set
    """
    unpicked_orders, num = CountUnpickedOrders(customer_set, 0 , interval = 0, return_type = 'name')
    bundle_names = []
    names = []
    res = {}
    #info = [[route, max(ftds), average(ftds), min(ftds), names],...,]
    for info in bundle_infos:
        bundle_names += info[4]
        if len(info[4]) == 1:
            customer = customer_set[info[4][0]]
            pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
            customer.platform_exp_cook_time = random.choice(pool)
            route = [[customer.name, 0, customer.store_loc, 0],[customer.name, 1, customer.location, 0]]
            o = Task(order_index, info[4][0], route, 'single', fee = customer.fee)
        else:
            route = []
            for node in info[0]:
                if node >= M:
                    customer_name = node - M
                    customer = customer_set[customer_name]
                    route.append([customer_name, 0, customer.store_loc, 0])
                else:
                    customer_name = node
                    customer = customer_set[customer_name]
                    route.append([customer_name, 1, customer.location, 0])
            fee = 0
            for customer_name in info[4]:
                fee += customer_set[customer_name].fee #주문의 금액 더하기.
                customer_set[customer_name].in_bundle_time = now_t
                pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
                customer_set[customer_name].platform_exp_cook_time = random.choice(pool)
            o = Task(order_index, info[4], route, 'bundle', fee = fee)
            o.olf_info = info
        o.average_ftd = info[2]
        res[order_index] = o
        #res.append(o)
        order_index += 1
    for index in platform_set.platform:
        order = platform_set.platform[index]
        if order.type == 'single':
            if order.customers[0] not in bundle_names and order.picked == False and customer_set[order.customers[0]].time_info[1] == None:
                res[order.index] = order
            else:
                pass
        else:
            if order.picked == False:
                #res.append(order)
                pass
    already_ordered_customer_names = []
    for index in res:
        already_ordered_customer_names += res[index].customers
    for index in platform_set.platform:
        already_ordered_customer_names += platform_set.platform[index].customers
    for customer_name in unpicked_orders:
        if divide_option == True:
            condition = customer_name not in already_ordered_customer_names
        else:
            condition = customer_name not in bundle_names + already_ordered_customer_names
        #if customer_name not in bundle_names + already_ordered_customer_names:
        if condition == True:
            names.append(customer_name)
            customer = customer_set[customer_name]
            if customer.time_info[1] == None:
                singleroute = [[customer.name , 0 , customer.store_loc,0],[customer.name, 1, customer.location, 0]]
                o = Task(order_index, [customer_name], singleroute, 'single', fee = customer.fee)
                #res.append(o)
                res[order_index] = o
                order_index += 1
                #print('추가 정보22 {}'.format(customer_name))
    return res


def PlatformOrderRevise2(bundle_infos, customer_set, order_index, platform_set, M = 1000, divide_option = False, now_t = 0, platform_exp_error = 1, unserved_bundle_order_break = False):
    """
    Construct unpicked_orders with bundled customer
    :param bundles: constructed bundles
    :param customer_set: customer list : [customer class,...,]
    :return: unserved customer set
    """
    unpicked_orders, num = CountUnpickedOrders(customer_set, 0 , interval = 0, return_type = 'name')
    bundle_names = []
    names = []
    res = {}
    #info = [[route, max(ftds), average(ftds), min(ftds), names],...,]
    for info in bundle_infos:
        bundle_names += info[4]
        if len(info[4]) == 1:
            customer = customer_set[info[4][0]]
            pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
            customer.platform_exp_cook_time = random.choice(pool)
            route = [[customer.name, 0, customer.store_loc, 0],[customer.name, 1, customer.location, 0]]
            o = Task(order_index, info[4][0], route, 'single', fee = customer.fee , parameter_info= None)
        else:
            route = []
            for node in info[0]:
                if node >= M:
                    customer_name = node - M
                    customer = customer_set[customer_name]
                    route.append([customer_name, 0, customer.store_loc, 0])
                else:
                    customer_name = node
                    customer = customer_set[customer_name]
                    route.append([customer_name, 1, customer.location, 0])
            fee = 0
            for customer_name in info[4]:
                fee += customer_set[customer_name].fee #주문의 금액 더하기.
                customer_set[customer_name].in_bundle_time = now_t
                pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
                customer_set[customer_name].platform_exp_cook_time = random.choice(pool)
            o = Task(order_index, info[4], route, 'bundle', fee = fee, parameter_info= info[7:10])
            o.olf_info = info
        o.average_ftd = info[2]
        res[order_index] = o
        #res.append(o)
        order_index += 1
    for index in platform_set.platform:
        order = platform_set.platform[index]
        if order.type == 'single':
            if order.customers[0] not in bundle_names and order.picked == False and customer_set[order.customers[0]].time_info[1] == None:
                res[order.index] = order
            else:
                pass
        else:
            if order.picked == False:
                #만약 겹치는 부분이 있다면, 이를 비교할 것.
                if unserved_bundle_order_break == True: #-> 기존에 발생한 번들과 신규 번들 사이에 겹침이 발생할 수 있음.
                    duplicate_customers = list(set(order.customers).intersection(set(bundle_names)))
                    if len(duplicate_customers) == 0:
                        res[order.index] = order
                    else:
                        #만약 기존 번들이 더 효과적이라면:
                        for new_order_index in res:
                            order = res[new_order_index]
                            if len(order.customers) > 1:
                                duplicate_customers = list(set(order.customers).intersection(set(bundle_names)))
                        #아니라면
                        for ct_name in order.customers:
                            if ct_name not in duplicate_customers:
                                #개별 고객으로 다시 넣어야 함.
                                customer = customer_set[ct_name]
                                pool = np.random.normal(customer.cook_info[1][0],customer.cook_info[1][1] * platform_exp_error, 1000)
                                customer.platform_exp_cook_time = random.choice(pool)
                                route = [[customer.name, 0, customer.store_loc, 0], [customer.name, 1, customer.location, 0]]
                                o = Task(order_index, [customer.name], route, 'single', fee=customer.fee,parameter_info=None)
                                o.average_ftd = 0
                                res[order_index] = o
                                order_index += 1
                        pass
                    #겹치는 부분이 발생한다면, 이를 삭제.
                    pass
                else:
                    res[order.index] = order
    already_ordered_customer_names = []
    for index in res:
        already_ordered_customer_names += res[index].customers
    for index in platform_set.platform:
        already_ordered_customer_names += platform_set.platform[index].customers
    for customer_name in unpicked_orders:
        if divide_option == True:
            condition = customer_name not in already_ordered_customer_names
        else:
            condition = customer_name not in bundle_names + already_ordered_customer_names
        #if customer_name not in bundle_names + already_ordered_customer_names:
        if condition == True:
            names.append(customer_name)
            customer = customer_set[customer_name]
            if customer.time_info[1] == None:
                singleroute = [[customer.name , 0 , customer.store_loc,0],[customer.name, 1, customer.location, 0]]
                o = Task(order_index, [customer_name], singleroute, 'single', fee = customer.fee)
                #res.append(o)
                res[order_index] = o
                order_index += 1
                #print('추가 정보22 {}'.format(customer_name))
    return res

def GenSingleOrder(order_index, customer, platform_exp_error = 1):
    pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
    customer.platform_exp_cook_time = random.choice(pool)
    route = [[customer.name, 0, customer.store_loc, 0], [customer.name, 1, customer.location, 0]]
    o = Task(order_index, customer.name, route, 'single', fee=customer.fee, parameter_info=None)
    return o

def GenBundleOrder(order_index, bundie_info, customer_set, now_t, M = 1000, platform_exp_error = 1):
    route = []
    for node in bundie_info[0]:
        if node >= M:
            customer_name = node - M
            customer = customer_set[customer_name]
            route.append([customer_name, 0, customer.store_loc, 0])
        else:
            customer_name = node
            customer = customer_set[customer_name]
            route.append([customer_name, 1, customer.location, 0])
    fee = 0
    for customer_name in bundie_info[4]:
        fee += customer_set[customer_name].fee  # 주문의 금액 더하기.
        customer_set[customer_name].in_bundle_time = now_t
        pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
        customer_set[customer_name].platform_exp_cook_time = random.choice(pool)
    o = Task(order_index, bundie_info[4], route, 'bundle', fee=fee, parameter_info=bundie_info[7:10])
    o.olf_info = bundie_info
    o.average_ftd = bundie_info[2]
    return o


def PlatformOrderRevise4(bundle_infos, customer_set, platform_set, now_t = 0, unserved_bundle_order_break = False, divide_option = True):
    """
    Construct unpicked_orders with bundled customer
    :param bundles: constructed bundles
    :param customer_set: customer list : [customer class,...,]
    :return: unserved customer set
    """
    order_indexs = []
    for index in platform_set.platform:
        order_indexs.append(index)
    order_index = 1
    if len(order_indexs) > 0:
        order_index = max(order_indexs) + 1
    #1 단건 주문 먼저 오더에 넣기
    added_single_customers = []
    res = {}
    for info in bundle_infos:
        if len(info[4]) == 1:
            customer = customer_set[info[4][0]]
            o = GenSingleOrder(order_index, customer)
            res[order_index] = o
            order_index += 1
            added_single_customers.append(customer.name)
    for order_index in platform_set.platform:
        order = platform_set.platform[order_index]
        if len(order.customers) == 1:
            res[order.index] = order
            added_single_customers += order.customers
    #2번들 처리
    for info in bundle_infos:
        if len(info[4]) > 1:
            o = GenBundleOrder(order_index, info, customer_set, now_t)
            o.old_info = info
            res[order_index] = o
    if unserved_bundle_order_break == False:
        for order_index in platform_set.platform:
            order = platform_set.platform[order_index]
            if len(order.customers) > 1:
                res[order.index] = order
    unpicked_orders, interval_orders = CountUnpickedOrders(customer_set, now_t , return_type = 'list')
    for customer_name in unpicked_orders:
        if customer_name not in added_single_customers:
            customer = customer_set[customer_name]
            o = GenSingleOrder(order_index, customer)
            res[order_index] = o
            order_index += 1
            added_single_customers.append(customer_name)
    if divide_option == True: #번들에 있는데, 개별 고객이 없는 경우를 추가
        in_bundle_customers = []
        single_order_customers = []
        for order_index in res:
            order = res[order_index]
            if order.type == 'bundle':
                in_bundle_customers += order.customers
            else:
                single_order_customers += order.customers
        for customer_name in in_bundle_customers:
            if customer_name not in single_order_customers:
                customer = customer_set[customer_name]
                o = GenSingleOrder(order_index, customer)
                res[order_index] = o
                order_index += 1
                single_order_customers.append(customer_name)
    return res


def ConsideredCustomer(platform_set, orders, unserved_order_break = False):
    """
    번들 구성에 고려될 수 있는 고객들을 선별함.
    @param platform_set: platform set list [class order, ...,]
    @param orders: customer set {[KY] customer name : [Value] class customer}
    @param unserved_order_break: T: 서비스 되지 않은 번들도 번들 구성에 고려/ F : 이미 구성된 번들은 고려하지 않음.
    @return: 번들 구성에 사용될 수 있는 고객 {[KY] customer name : [Value] class customer}
    """
    rev_order = {}  # 아직 서비스 되지 않은 고객 + 플렛폼에 있으나, 아직 번들로 구성되지 않은 주문 [KY] 고객 이름
    except_names = []
    #input('확인1 {}'.format(platform_set.platform))
    for index in platform_set.platform:
        #input('확인2 {}'.format(index))
        order = platform_set.platform[index]
        if order.type == 'single':
            if order.picked == False and orders[order.customers[0]].time_info[1] == None:
                rev_order[order.customers[0]] = orders[order.customers[0]]
            else: #already picked customer
                pass
        else: #번들인 경우
            if order.picked == False:
                if unserved_order_break == False:
                    except_names += order.customers  # todo: 기존에 구성된 번들은 해체되지 않음.
                else:
                    pass
            else: #이미 선택된 주문의 경우
                pass
    for customer_name in orders:
        customer = orders[customer_name]
        if customer.time_info[1] == None and customer_name not in list(rev_order.keys()) + except_names:
            rev_order[customer_name] = customer
    print1 = []
    for order_name in rev_order:
        order = rev_order[order_name]
        #print1 += order.customers
        print1.append(order.name)
    print2 = []
    for customer_name in orders:
        customer = orders[customer_name]
        if customer.time_info[1] != None:
            print2.append(customer.name)
    print('번들 대상 고객 {}'.format(print1))
    print('실려있는 고객 {}'.format(print2))
    return rev_order


def ResultPrint(name, customers, speed = 1, riders = None):
    rider_income_var = None
    if riders != None:
        riders_incomes = []
        for rider_name in riders:
            rider = riders[rider_name]
            riders_incomes.append(rider.income)
        rider_income_var = np.var(riders_incomes)
    served_customer = []
    TLT = []
    FLT = []
    MFLT = []
    for customer_name in customers:
        customer = customers[customer_name]
        if customer.time_info[3] != None:
            lt = customer.time_info[3] - customer.time_info[0]
            flt = customer.time_info[3] - customer.time_info[2]
            mflt = distance(customer.store_loc, customer.location)/speed
            TLT.append(lt)
            FLT.append(flt)
            MFLT.append(mflt)
    customer_lead_time_var = np.var(TLT)
    try:
        served_ratio = round(len(TLT)/len(customers),2)
        av_TLT = round(sum(TLT)/len(TLT),2)
        av_FLT = round(sum(FLT)/len(FLT),2)
        av_MFLT = av_FLT - round(sum(MFLT)/len(MFLT),2)
        print('시나리오 명 {} 전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'.format(name, len(customers), len(TLT),served_ratio,av_TLT,
                                                                             av_FLT, av_MFLT))
        return [len(customers), len(TLT),served_ratio,av_TLT,av_FLT, av_MFLT, round(sum(MFLT)/len(MFLT),2), rider_income_var,customer_lead_time_var]
    except:
        print('TLT 수:  {}'.format(len(TLT)))
        return None

def CountIdleRiders(riders, now_t , interval = 10, return_type = 'class'):
    """
    return idle rider
    :param riders: rider list : [rider class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: idle_riders, mu2(future generated rider)
    """
    idle_riders = []
    interval_riders = []
    for rider_name in riders:
        #Count current idle rider
        rider = riders[rider_name]
        if len(rider.resource.users) == 0:
            if return_type == 'class':
                idle_riders.append(rider)
            elif return_type == 'name':
                idle_riders.append(rider.name)
            else:
                pass
        #count rider occurred from (now_t - interval, now)
        if now_t- interval <= rider.start_time < now_t:
            interval_riders.append(rider.name)
    return idle_riders, len(interval_riders)