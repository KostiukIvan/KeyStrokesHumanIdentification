
from pynput import keyboard
import time

def get_press_release_data():
    press_events = []
    def on_press(key):
        if type(key) is keyboard.KeyCode:
          value = ord(key.char)   
          press_events.append((value, time.time()))
        else:
           pass
           # TODO: all non-alphanumeric keyboard key are skipped. Temporary
        
    release_events = []
    def on_release(key):
        if type(key) is keyboard.KeyCode:
          value = ord(key.char)   
          release_events.append((value, time.time()))
        else:
           pass
           # TODO: all non-alphanumeric keyboard key are skipped. Temporary
        if key == keyboard.Key.esc:
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        
    return press_events, release_events

