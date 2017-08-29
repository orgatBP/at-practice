
import random 
import time    

DEBUG = False 

N = 1000     # number of elements in list
list1 = []   # list of integer elements

for i in range(0, N):
    list1.append(random.randint(0, N-1))



def print_timing(func):
    def wrapper(*arg):
        t1 = time.clock()
        res = func(*arg)
        t2 = time.clock()
        print '%s took %0.3fms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper


@print_timing
def adaptive_merge_sort(list2):
    """adaptive merge sort, built into Python since version 2.3"""
    list2.sort()

@print_timing
def bubble_sort(list2):
    swap_test = False
    for i in range(0, len(list2) - 1):
        for j in range(0, len(list2) - i - 1):
            if list2[j] > list2[j + 1]:
                list2[j], list2[j + 1] = list2[j + 1], list2[j]  # swap
            swap_test = True
        if swap_test == False:
            break


@print_timing
def selection_sort(list2):
    for i in range(0, len (list2)):
        min = i
        for j in range(i + 1, len(list2)):
            if list2[j] < list2[min]:
                min = j
        list2[i], list2[min] = list2[min], list2[i] 
      

@print_timing
def insertion_sort(list2):
    for i in range(1, len(list2)):
        save = list2[i]
        j = i
        while j > 0 and list2[j - 1] > save:
            list2[j] = list2[j - 1]
            j -= 1
        list2[j] = save
  

@print_timing
def quick_sort(list2):
    quick_sort_r(list2, 0, len(list2) - 1)


def quick_sort_r(list2 , first, last):
    if last > first:
        pivot = partition(list2, first, last)
        quick_sort_r(list2, first, pivot - 1)
        quick_sort_r(list2, pivot + 1, last)


def partition(list2, first, last):
    sred = (first + last)/2

    if list2[first] > list2 [sred]:
        list2[first], list2[sred] = list2[sred], list2[first]  # swap

    if list2[first] > list2 [last]:
        list2[first], list2[last] = list2[last], list2[first]  # swap

    if list2[sred] > list2[last]:
        list2[sred], list2[last] = list2[last], list2[sred]    # swap
    list2 [sred], list2 [first] = list2[first], list2[sred]    # swap

    pivot = first
    i = first + 1
    j = last
  
    while True:
        while i <= last and list2[i] <= list2[pivot]:
            i += 1
        while j >= first and list2[j] > list2[pivot]:
            j -= 1
        if i >= j:
            break
        else:
            list2[i], list2[j] = list2[j], list2[i]  # swap
    list2[j], list2[pivot] = list2[pivot], list2[j]  # swap
    return j

#www.iplaypy.com
@print_timing
def heap_sort(list2):
    first = 0
    last = len(list2) - 1
    create_heap(list2, first, last)
    for i in range(last, first, -1):
        list2[i], list2[first] = list2[first], list2[i] 
        establish_heap_property (list2, first, i - 1)


def create_heap(list2, first, last):
    i = last/2

    while i >= first:
        establish_heap_property(list2, i, last)
        i -= 1


def establish_heap_property(list2, first, last):
    while 2 * first + 1 <= last:

        k = 2 * first + 1

        if k < last and list2[k] < list2[k + 1]:
            k += 1

        if list2[first] >= list2[k]:
            break

        list2[first], list2[k] = list2[k], list2[first]  

        first = k


@print_timing
def merge_sort(list2):
    merge_sort_r(list2, 0, len(list2) -1)


def merge_sort_r(list2, first, last):
    if first < last:

        sred = (first + last)/2

        merge_sort_r(list2, first, sred)
        merge_sort_r(list2, sred + 1, last)
        merge(list2, first, last, sred)


def merge(list2, first, last, sred):
    helper_list = []

    i = first

    j = sred + 1

    while i <= sred and j <= last:
        if list2 [i] <= list2 [j]:
            helper_list.append(list2[i])
            i += 1
        else:
            helper_list.append(list2 [j])
            j += 1
    while i <= sred:
        helper_list.append(list2[i])
        i +=1
    while j &l
2000

t;= last:
        helper_list.append(list2[j])
        j += 1

    for k in range(0, last - first + 1):
        list2[first + k] = helper_list [k]


def print10(list2):
    for k in range(10):
        print list2[k],
    print



if __name__ == "__main__" :
    print "timing 7 sorting algorithms with a list of 1000 integers:"
   
    list2 = list(list1)

    adaptive_merge_sort(list2)

    if DEBUG:
        print10(list2)
    list2 = list(list1)
    bubble_sort(list2)
    if DEBUG:
        print10(list2) 
   list2 = list(list1)
    heap_sort(list2)
    if DEBUG:
        print10(list2)
    list2 = list(list1)
    insertion_sort(list2)
    if DEBUG:
        print10(list2)
    list2 = list(list1)
    merge_sort(list2)
    if DEBUG:
        print10(list2)
    list2 = list(list1)
    quick_sort(list2)
    if DEBUG:
        print10(list2)
    list2 = list(list1)
    selection_sort(list2)
    if DEBUG:
        print10(list2)
    # final test
    list2 = list(list1)
    if DEBUG:
        print "final test: ",
        print10(list2)

   
