from machine.car import drive
from machine.tv import watch

drive()
watch()

################################################
print('='*80)
# from machine import car
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()

print('='*80)
from machine.test import car, tv

car.drive()
tv.watch()
