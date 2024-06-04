from pyModbusTCP.client import ModbusClient

from pyModbusTCP.server import ModbusServer
import threading
import time



class CLP:
    def __init__(self, host="127.0.0.1", port=520, unit_id=1, auto_open=True, stop_signal=None):
        print(host)
        self.client = ModbusClient(host, port, unit_id=unit_id, auto_open=True, auto_close=False, timeout=5, debug=False)
        self.client.open()
        self.coil_callbacks = {}
        self.has_stopped = stop_signal

        if not self.client.is_open:
            self.has_stopped.set()
            raise ConnectionRefusedError("Cant communicate with clp")
        # Start a background thread to monitor coil states
        self.thread = threading.Thread(target=self._monitor_coils, daemon=True)
        self.thread.start()
    
    def only_if_connected(foo):
        def magic(self, *args, **kwargs) :
            if self.client.is_open:
                return foo( self, *args, **kwargs )
            print("CLP is not conected.")
        return magic

    @only_if_connected
    def write_group_data(self, start_address, array, group_size=20, reg=False):
        print(f"Atualizando CLP com {len(array)} informações")
        subarrays = [array[i:i+group_size] for i in range(0, len(array), group_size)]

        # Enviar cada subarray para os registradores correspondentes
        for i, subarray in enumerate(subarrays):
            address = start_address + i * group_size
            #print(address, subarray)
            if reg:
                d = [int(i*100) for i in subarray]
                # print(d)
                self.client.write_multiple_registers(address, d)
            else:
                R = False
                while not R:
                    R = self.client.write_multiple_coils(address, subarray)
                    if not R:
                        print("Fail to Write at:", address, " to", address+len(subarray))
    
    def register(self, coil_addr, event_type, callback_func):
        if coil_addr not in self.coil_callbacks:
            self.coil_callbacks[coil_addr] = []

        self.coil_callbacks[coil_addr].append((event_type, callback_func))

    def _monitor_coils(self):
        coil_states = {}
        fails = 0
        while self.client.is_open and not self.has_stopped.is_set():
            for coil_addr in self.coil_callbacks.keys():
                result = self.client.read_coils(coil_addr, 1)
                if result is None:
                    # print("Cant read.")
                    # fails+=1
                    time.sleep(2)
                    continue
                # fails=0
                # if not result.isError():
                current_state = result[0]  #>
                if coil_addr in coil_states:
                    previous_state = coil_states[coil_addr]
                    if current_state != previous_state:
                        self._trigger_callbacks(coil_addr, previous_state, current_state)
                coil_states[coil_addr] = current_state

            time.sleep(1)  # Adjust the polling interval as needed
        else:
            self.has_stopped.set()
            raise ConnectionRefusedError(f"Cant communicate with CLP in {fails} attempts.")
            # return

    def _trigger_callbacks(self, coil_addr, previous_state, current_state):
        for event_type, callback_func in self.coil_callbacks[coil_addr]:
            if event_type == 'RISING' and previous_state == False and current_state == True:
                callback_func(self, True)
            elif event_type == 'FALLING' and previous_state == True and current_state == False:
                callback_func(self, False)
            elif event_type == 'CHANGE':
                callback_func(self, current_state)
