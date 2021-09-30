# -*- coding: utf-8 -*-
import math
import random
import csv
import numpy.random
import Class as Class
import time

def distance(p1, p2):
    """
    Calculate 4 digit rounded euclidean distance between p1 and p2
    :param p1:
    :param p2:
    :return: 4 digit rounded euclidean distance between p1 and p2
    """
    euc_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return round(euc_dist,4)


def RouteTime(orders, route, M = 1000, speed = 1, uncertainty = False, error = 1):
    """
    Time to move the route with speed
    :param orders: order in route
    :param route: seq
    :param speed: rider speed
    :return: time : float
    """
    time = 0
    locs = {}
    names = []
    if type(orders) == dict:
        for order_name in orders:
            locs[order_name + M] = [orders[order_name].store_loc, 'store', orders[order_name].time_info[6]]
            locs[order_name] = [orders[order_name].location, 'customer', orders[order_name].time_info[7]]
            names += [order_name + M, order_name]
    elif type(orders) == list:
        for order in orders:
            locs[order.name + M] = [order.store_loc, 'store', order.time_info[6]]
            locs[order.name] = [order.location, 'customer', order.time_info[7]]
            names += [order.name + M, order.name]
    else:
        input('Error')
    #print('고려 대상들{} 경로{}'.format(list(locs.keys()), route))
    for index in range(1,len(route)):
        bf = route[index-1]
        bf_loc = locs[bf][0]
        af = route[index]
        #print(1, bf,af,time)
        af_loc = locs[af][0]
        time += distance(bf_loc,af_loc)/speed + locs[af][2]
        if af > M:
            for order in orders:
                if order.name == af - M:
                    target = order
            if uncertainty == True and target.cook_info[0] == 'uncertainty': #todo : 추가 시간이 발생할 수 있음을 반영
                pool = numpy.random.normal(target.cook_info[1][0], target.cook_info[1][1]*error, 1000)
                exp_cook_time = random.choice(pool)
                if exp_cook_time > time:
                    time += exp_cook_time - time
    return round(time,4)


def FLT_Calculate(customer_in_order, customers, route, p2, except_names , M = 1000, speed = 1, now_t = 0, uncertainty = False, exp_error = 1):
    """
    Calculate the customer`s Food Delivery Time in route(bundle)

    :param orders: customer order in the route. type: customer class
    :param route: customer route. [int,...,]
    :param p2: allowable FLT increase
    :param speed: rider speed
    :return: Feasiblity : True/False, FLT list : [float,...,]
    """
    names = []
    for order in customer_in_order:
        if order.name not in names:
            names.append(order.name)
    ftds = []
    for order_name in names:
        if order_name not in except_names:
            #rev_p2 = p2
            rev_p2 = customers[order_name].p2
            if customers[order_name].time_info[2] != None:
                last_time = now_t - customers[order_name].time_info[2] #이미 음식이 실린 후 지난 시간
                #rev_p2 = p2 - last_time
                rev_p2 = customers[order_name].min_FLT - last_time
            try:
                s = route.index(order_name + M)
                e = route.index(order_name)
                try:
                    ftd = RouteTime(customer_in_order, route[s: e + 1], speed=speed, M=M, uncertainty=uncertainty, error = exp_error)
                except:
                    ftd = 1000
            except:
                ftd = 0
                print('경로 {}'.format(route))
                print('인덱스 에러 발생 현재 고객 이름 {} 경로 고객들 {} 경로 {}'.format(order_name, names, route))
            if ftd > rev_p2:
                return False, []
            else:
                ftds.append(ftd)
    return True, ftds


def RiderGenerator(env, Rider_dict, Platform, Store_dict, Customer_dict, capacity = 3, speed = 1, working_duration = 120, interval = 1, runtime = 1000, gen_num = 10, history = None, freedom = True, score_type = 'simple', wait_para = False, uncertainty = False, exp_error = 1, exp_WagePerHr = 9000):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env: simpy environment
    :param Rider_dict: 플랫폼에 있는 라이더들 {[KY]rider name : [Value]class rider, ...}
    :param rider_name: 라이더 이름 int+
    :param Platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param Store_dict: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param Customer_dict:발생한 고객들 {[KY]customer name : [Value]class customer, ...}
    :param working_duration: 운행 시작 후 운행을 하는 시간
    :param interval: 라이더 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    :param gen_num: 생성 라이더 수
    """
    rider_num = 0
    while env.now <= runtime and rider_num <= gen_num:
        single_rider = Class.Rider(env,rider_num,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, capacity = capacity, freedom=freedom, order_select_type = score_type, wait_para =wait_para, uncertainty = uncertainty, exp_error = exp_error)
        single_rider.exp_wage = exp_WagePerHr
        Rider_dict[rider_num] = single_rider
        #print('T {} 라이더 {} 생성'.format(int(env.now), rider_num))
        print('라이더 {} 생성. T {}'.format(rider_num, int(env.now)))
        if history != None:
            #next = history[rider_num + 1] - history[rider_num]
            next = history[rider_num]
            yield env.timeout(next)
        else:
            yield env.timeout(interval)
        rider_num += 1


def RiderGeneratorByCSV(env, csv_dir, Rider_dict, Platform, Store_dict, Customer_dict, working_duration = 120, exp_WagePerHr = 9000 ,input_speed = None, input_capacity = None):
    """
    Generate the rider until t <= runtime and rider_num<= gen_num
    :param env: simpy environment
    :param Rider_dict: 플랫폼에 있는 라이더들 {[KY]rider name : [Value]class rider, ...}
    :param rider_name: 라이더 이름 int+
    :param Platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param Store_dict: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param Customer_dict:발생한 고객들 {[KY]customer name : [Value]class customer, ...}
    :param working_duration: 운행 시작 후 운행을 하는 시간
    :param interval: 라이더 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    :param gen_num: 생성 라이더 수
    """
    datas = ReadCSV(csv_dir, interval_index = 1)
    interval_index = len(datas[0]) - 1
    for data in datas:
        name = data[0]
        if input_speed == None:
            speed = data[2]
        else:
            speed = input_speed
        if input_capacity == None:
            capacity = data[4]
        else:
            capacity = input_capacity
        freedom = data[5]
        order_select_type = data[6]
        wait_para = data[7]
        uncertainty = data[8]
        exp_error = data[9]
        single_rider = Class.Rider(env,name,Platform, Customer_dict,  Store_dict, start_time = env.now ,speed = speed, end_t = working_duration, \
                                   capacity = capacity, freedom=freedom, order_select_type = order_select_type, wait_para =wait_para, uncertainty = uncertainty, exp_error = exp_error)
        single_rider.exp_wage = exp_WagePerHr
        Rider_dict[name] = single_rider
        interval = data[interval_index]
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))



def GenerateStoreByCSV(env, csv_dir, platform,Store_dict):
    datas = ReadCSV(csv_dir)
    for data in datas:
        #['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
        name = data[0]
        loc = [data[1], data[2]]
        order_ready_time = data[3]
        capacity = data[4]
        slack = data[5]
        store = Class.Store(env, platform, name, loc=loc, order_ready_time=order_ready_time, capacity=capacity, print_para=False, slack = slack)
        Store_dict[name] = store


def Ordergenerator(env, orders, stores, max_range = 50, interval = 5, runtime = 100, history = None, p2 = 15, p2_set = False, speed = 4, cooking_time = [2,5], cook_time_type = 'random'):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    name = 0
    while env.now < runtime:
        #process_time = random.randrange(1,5)
        #input_location = [36,36]
        if history == None:
            input_location = random.sample(list(range(max_range)),2)
            store_num = random.randrange(0, len(stores))
        else:
            input_location = history[name][2]
            store_num = history[name][1]
            interval = history[name + 1][0] - history[name][0]
        if cook_time_type == 'random':
            cook_time = random.randrange(cooking_time[0],cooking_time[1])
        else:
            pool = numpy.random.normal(cooking_time[0],cooking_time[1], 1000)
            cook_time = round(random.choice(pool),4)
        if cook_time < 0:
            input('조리 시간 음수 {}/ 생성 정보 {}'.format(cook_time,cooking_time))
        order = Class.Customer(env, name, input_location, store=store_num, store_loc=stores[store_num].location, p2=p2, cooking_time = cook_time, cook_info = [cook_time_type, cooking_time])
        #input('주문 {} 정보 {}'.format(order.name, order.cook_info))
        if p2_set == True:
            if type(p2) == list:
                selected_p2 = random.choices(population=p2[0],weights=p2[1],k=1)
                selected_p2 = selected_p2[0]
                #input('test {} {} {}'.format(selected_p2, order.distance, speed))
                order.p2 = selected_p2 * order.distance / speed
                order.min_FLT = order.p2
            else:
                order.p2 = p2 * (order.distance / speed)
                order.min_FLT = order.p2
                #input('p2 {} / dist {}  order.p2 {}'.format(p2, order.p2))
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        yield env.timeout(interval)
        #print('현재 {} 플랫폼 주문 수 {}'.format(int(env.now), len(platform)))
        name += 1


def ReadCSV(csv_dir, interval_index = None):
    raw_datas = []
    datas = []
    #csv 파일 읽기
    f = open(csv_dir+'.csv','r')
    rdr = csv.reader(f)
    for line in rdr:
        raw_datas.append(line)
    f.close()
    for raw_data in raw_datas[1:]:
        tem = []
        for info in raw_data:
            try:
                num = float(info)
                if round(num) == num:
                    tem.append(int(num))
                else:
                    tem.append(num)
            except:
                tem.append(str(info))
        datas.append(tem)
    if interval_index != None:
        for index in range(1, len(datas)):
            interval = datas[index][interval_index] - datas[index - 1][interval_index]
            datas[index - 1].append(interval)
        datas[-1].append(0)
    return datas

def OrdergeneratorByCSV(env, csv_dir, orders, stores):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    #CSV 파일 읽기
    datas = ReadCSV(csv_dir, interval_index = 1)
    interval_index = len(datas[0]) - 1
    for data in datas:
        #[customer.name, customer.time_info[0], customer.location[0],customer.location[1], customer.store, customer.store_loc[0],customer.store_loc[1], customer.p2, customer.cook_time, customer.cook_info[0], customer.cook_info[1][0], customer.cook_info[1][1]]
        name = data[0]
        gen_t = data[1]
        input_location = [data[2],data[3]]
        store_num = data[4]
        store_loc = [data[5], data[6]]
        p2 = data[7]
        cook_time = data[8]
        cook_time_type = data[9]
        cooking_time = [data[10], data[11]]
        order = Class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
                               cooking_time=cook_time, cook_info=[cook_time_type, cooking_time])
        orders[name] = order
        stores[store_num].received_orders.append(orders[name])
        interval = data[interval_index]
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass



def ReadRiderData(env, rider_data, Platform, Rider_dict, Customer_dict, Store_dict):
    #저장된 txt 데이터를 읽고, 그에 따라서 인스턴스 생성
    #rider_data = [name, start_loc, gen_time, ExpectWagePerHr]
    #order_data = [name, store_num, loc , gen_time]
    #stroe_data = [name, capacity, loc]
    f = open(rider_data + ".txt", 'r')
    lines = f.readlines()
    rider_num = 1
    for line in lines[1:]:
        line.split('')
        """
        single_rider = Class.Rider(env, rider_num, Platform, Customer_dict, Store_dict, start_time = env.now, speed = speed, end_t = working_duration,\
                                   capacity = capacity, freedom = freedom, order_select_type = score_type, wait_para = wait_para, uncertainty = uncertainty\
                                   exp_error = exp_error)
        single_rider.exp_wage = exp_WagePerHr      
        Rider_dict[rider_num] = single_rider          
        """
        interval = 1
        rider_num += 1
        yield env.timeout(interval)

    f.close()
    return None


def UpdatePlatformByOrderSelection(platform, order_index):
    """
    선택된 주문과 겹치는 고객을 가지는 주문이 플랫폼에 존재한다면, 해당 주문을 삭제하는 함수.
    @param platform: class platform
    @param order_index: 라이더가 선택한 주문.
    """
    delete_order_index = []
    order = platform.platform[order_index]
    for order_index1 in platform.platform:
        compare_order = platform.platform[order_index1]
        if order_index != order_index1:
            duplicate_customers = list(set(order.customers).intersection(compare_order.customers))
            if len(duplicate_customers) > 0:
                delete_order_index.append(compare_order.index)
    print('지우기 이전 {}'.format(platform.platform.keys()))
    print('삭제 index {}/ 삭제할 {}'.format(order_index, delete_order_index))
    for order_index in delete_order_index:
        del platform.platform[order_index]
    del platform.platform[order_index]
    print('지우기 이후 {}'.format(platform.platform.keys()))

def ActiveRiderCalculator(rider, t_now = 0, option = None, interval = 5):
    """
    현재 라이더가 새로운 주문을 선택할 수 있는지 유/무를 계산.
    @param rider: class rider
    @return: True/ False
    """
    if t_now <= rider.end_t :
        if option == None:
            if len(rider.picked_orders) < rider.max_order_num:
                print('문구1/ 라이더 {} / 현재 OnHandOrder# {} / 최대 주문 수{} / 예상 선택 시간 {} / 다음 interval 시간 {}'.format(rider.name,len(rider.picked_orders), rider.max_order_num, round(rider.next_select_t,2), t_now + interval))
                return True
        else: #w 계산 시 고려 필요
            if len(rider.picked_orders) <= rider.max_order_num and t_now <= rider.next_select_t <= t_now + interval:
                print('문구2/ 라이더 {} / 현재 OnHandOrder# {} / 최대 주문 수{} / 예상 선택 시간 {} / 다음 interval 시간 {}'.format(rider.name,len(rider.picked_orders), rider.max_order_num, round(rider.next_select_t,2), t_now + interval))
                return True
    else:
        return False


def WillingtoWork(rider, t_now):
    """
    시간당 수익이 희망 시간당 수익보다 작은 경우의 그 양
    max(희망 시간당 수익 - 현재 라이더의 시간당 수익 ,0)
    @param rider: class rider
    @return: max(희망 시간당 수익 - 현재 라이더의 시간당 수익 ,0)
    """
    current_wagePerHr  = rider.income/((t_now - rider.gen_time)/60)
    if current_wagePerHr >= rider.exp_wage: #임금이 자신의 허용 범위보다 작다면
        return current_wagePerHr
    else:
        return 0


def ForABundleCount(route_info):
    B = []
    num_bundle_customer = 0
    bundle_start = 0
    b = 0
    for node in route_info:
        if node[1] == 0:
            store_index = route_info.index(node)
            for node2 in route_info[store_index:]:
                if node[0] == node2[0] and node2[1] == 1:
                    customer_index = route_info.index(node2)
                    if store_index + 1 < customer_index:
                        num_bundle_customer += 1
                        if store_index == bundle_start + 1:
                            #print('A', bundle_start, store_index, customer_index)
                            b += 1
                        else:
                            #print('B',bundle_start, store_index, customer_index)
                            B.append(b)
                            b = 0
                    break
            bundle_start = store_index
    return B, num_bundle_customer

def ResultSave(Riders, Customers, title = 'Test', sub_info = 'None', type_name = 'A'):
    tm = time.localtime(time.time())
    sub = ['Day {} Hr{}Min{}Sec{}/ SUB {} '.format(tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,sub_info)]
    rider_header = ['라이더 이름', '서비스 고객수', '주문 탐색 시간','선택한 번들 수','번들로 서비스된 고객 수','라이더 수익','음식점 대기시간','대기시간_번들','대기시간_단건주문','주문 선택 간격','경로']
    rider_infos = [sub,rider_header]
    for rider_name in Riders:
        rider = Riders[rider_name]
        if len(rider.bundle_store_wait) > 0:
            bundle_store_wait = round(sum(rider.bundle_store_wait) / len(rider.bundle_store_wait), 2)
        else:
            bundle_store_wait = 0
        if len(rider.single_store_wait) > 0:
            single_store_wait = round(sum(rider.single_store_wait) / len(rider.single_store_wait), 2)
        else:
            single_store_wait = None
        if type_name == 'A':
            bundle_num, num_bundle_customer = ForABundleCount(rider.visited_route)
            #input('라이더 {} : 번들 정보 {} : 번들 고객 수 {}'.format(rider.name, bundle_num,num_bundle_customer))
            rider.b_select = round(num_bundle_customer/2.5,2)
            rider.num_bundle_customer = num_bundle_customer
        decision_moment = []
        for time_index in range(1,len(rider.decision_moment)):
            decision_interval = rider.decision_moment[time_index] - rider.decision_moment[time_index - 1]
            decision_moment.append(decision_interval)
        #print('주문간격 시점 데이터 {}'.format(decision_moment))
        decision_moment = round(sum(decision_moment) / len(decision_moment), 2)
        #print('평균 주문간격{}'.format(decision_moment))
        info = [rider_name, len(rider.served), rider.idle_time, rider.b_select,rider.num_bundle_customer, int(rider.income), round(rider.store_wait,2) ,bundle_store_wait,single_store_wait,decision_moment,rider.visited_route]
        rider_infos.append(info)
    customer_header = ['고객 이름', '생성 시점', '라이더 선택 시점','가게 출발 시점','고객 도착 시점','가게 도착 시점','음식조리시간','음식 음식점 대기 시간','라이더 가게 대기시간1','라이더 가게 대기시간2','수수료', '수행 라이더 정보', '직선 거리','p2(민감정도)','번들여부','조리시간','기사 대기 시간','번들로 구성된 시점']
    customer_infos = [sub, customer_header]
    for customer_name in Customers:
        customer = Customers[customer_name]
        wait_t = None
        try:
            wait_t = customer.ready_time - customer.time_info[8] #음식이 준비된 시간 - 가게에 도착한 시간.
        except:
            pass
        info = [customer_name] + customer.time_info[:4] +[customer.time_info[8]]+[customer.cook_time]+ [customer.food_wait, customer.rider_wait]+[wait_t, customer.fee,customer.who_serve, customer.distance, customer.p2, customer.inbundle,customer.cook_time, customer.rider_wait,customer.in_bundle_time]
        customer_infos.append(info)
    f = open(title + "riders.txt", 'a')
    for info in rider_infos:
        count = 0
        for ele in info:
            data = ele
            if type(ele) != str:
                data = str(ele)
            f.write(data)
            f.write(';')
            count += 1
            if count == len(info):
                f.write('\n')
    f.close()
    f = open(title + "customers.txt", 'a')
    for info in customer_infos:
        count = 0
        for ele in info:
            data = ele
            if type(ele) != str:
                data = str(ele)
            f.write(data)
            f.write(';')
            count += 1
            if count == len(info):
                f.write('\n')
    f.close()


def SaveInstanceAsCSV(Rider_dict, Orders,Store_dict, instance_name = '' ):
    #시나리오 저장
    rider_header = ['name', 'start time', 'speed', 'end', 'capacity', 'freedom', 'order_select_type', 'wait_para', 'uncertainty', 'exp_error']
    saved_rider_infos = []
    saved_rider_infos.append(rider_header)
    for rider_name in Rider_dict:
        rider = Rider_dict[rider_name]
        tem = [rider.name, rider.start_time, rider.speed, rider.end_t, rider.capacity, rider.freedom, rider.order_select_type, rider.wait_para, rider.uncertainty, rider.exp_error]
        tem1 = []
        for info in tem:
            #tem1.append(';')
            tem1.append(info)
            #tem1.append(';')
        saved_rider_infos.append(tem)
    customer_header = ['name','gen t', 'loc_x', 'loc_y', 'store', 'store_x', 'store_y', 'p2', 'cook_time', 'cook_time_type','cook_time_s','cook_time_e',]
    saved_customer_infos = []
    saved_customer_infos.append(customer_header)
    for customer_name in Orders:
        customer = Orders[customer_name]
        tem = [customer.name, customer.time_info[0], customer.location[0],customer.location[1], customer.store, customer.store_loc[0],customer.store_loc[1], customer.p2, customer.cook_time, customer.cook_info[0], customer.cook_info[1][0], customer.cook_info[1][1]]
        saved_customer_infos.append(tem)
    store_header = ['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
    saved_store_infos = []
    saved_store_infos.append(store_header)
    for store_name in Store_dict:
        store = Store_dict[store_name]
        tem = [store.name,store.location[0],store.location[1], store.order_ready_time, store.capacity, store.slack]
        saved_store_infos.append(tem)
    #txt로 저장
    file_name = ['rider_infos', 'customer_infos', 'store_infos']
    name_index = 0
    for infos in [saved_rider_infos, saved_customer_infos, saved_store_infos]:
        #f = open("TEST" + file_name[name_index] + ".txt", 'w')
        f = open("Instance" + file_name[name_index] + instance_name +".csv", 'w', newline = '')
        wr = csv.writer(f)
        for info in infos:
            #f.write(str(info) + '\n')
            wr.writerow(info)
        f.close()
        name_index += 1