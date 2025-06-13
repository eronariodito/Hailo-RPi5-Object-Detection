from gpiozero import CPUTemperature
from time import sleep, strftime, time
from hailo_platform import Device
import psutil

def temp_ram_measurement(file, target_name, duration, time_interval):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == target_name:
            pid = proc.info['pid']

    cpu = CPUTemperature()

    device_infos = Device.scan()
    targets = [Device(di) for di in device_infos]

    start_time = time()

    with open(file, "a", buffering=1) as log:
        log.write(target_name  + "\n")
        while (time() - start_time < duration):
            mem = psutil.virtual_memory()
            temp_cpu = cpu.temperature
            for target in targets:
                temp_hailo = target.control.get_chip_temperature().ts0_temperature

            used_ram = mem.used / (1024 ** 2)

            process = psutil.Process(pid).memory_info().rss / (1024 ** 2)
            cpu_util = psutil.cpu_percent()
            
            log.write("{0},{1},{2:.1f},{3:.2f},{4:.2f}, {5:.2f}\n".format(strftime("%Y-%m-%d %H:%M:%S"),str(temp_cpu), (temp_hailo), (used_ram), (process),(cpu_util)))

            log.flush()
            sleep(time_interval)

    print(f"Measurement saved to {file}")