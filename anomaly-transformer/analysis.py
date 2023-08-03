from modules.preprocessing import plot_data



def main():
    signals_first = [
        'machine_nameKuka Robot_apparent_power',
        'machine_nameKuka Robot_current',
        'machine_nameKuka Robot_export_reactive_energy',
        'machine_nameKuka Robot_frequency',
        'machine_nameKuka Robot_import_active_energy',
        'machine_nameKuka Robot_phase_angle', 'machine_nameKuka Robot_power',
        'machine_nameKuka Robot_power_factor',
        'machine_nameKuka Robot_reactive_power',
        'machine_nameKuka Robot_voltage']
    signals_second = [
        "sensor_id1_AngY",
        "sensor_id2_AngX",
        "sensor_id5_AngY",
        "sensor_id4_AccZ",
        "sensor_id4_AngX",
        "machine_nameKuka Robot_power"]
    freq = 10
    path = './data/normal'
    plot_data(path, freq, signals_first)
    plot_data(path, freq, signals_second)
    
if __name__=="__main__":
    main()