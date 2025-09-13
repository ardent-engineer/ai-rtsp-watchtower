import pygame
from PyQt5.QtCore import QTimer

class AlarmHandler:
    def __init__(self, sound_paths, cooldown_ms, grey_zone_ms, black_zone_ms):
        """
        Initializes the AlarmHandler with configuration settings.
        
        Args:
            sound_paths (list): List of file paths for the alarm sounds.
            cooldown_ms (int): Cooldown period in milliseconds.
            grey_zone_ms (int): Grey zone duration in milliseconds.
            black_zone_ms (int): Black zone duration in milliseconds.
        """
        pygame.mixer.init()
        
        self.alarm_active = False
        self.cooldown_active = False
        self.grey_zone_active = False

        self.grey_zone_time = grey_zone_ms
        self.grey_zone_timer = QTimer()
        self.grey_zone_timer.setSingleShot(True)
        self.grey_zone_timer.timeout.connect(self.grey_zone_trigger)

        self.black_zone_active = False
        self.black_zone_time = black_zone_ms
        self.black_zone_timer = QTimer()
        self.black_zone_timer.setSingleShot(True)
        self.black_zone_timer.timeout.connect(self.black_zone_trigger)

        self.cooldown_time = cooldown_ms
        self.cooldown_timer = QTimer()
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self.end_cooldown)

        self.channel = pygame.mixer.Channel(0)
        self.sound = [pygame.mixer.Sound(p) for p in sound_paths]
        self.current_p = None

    def grey_zone_trigger(self):
        self.grey_zone_active = False
        self.black_zone_active = True
        self.black_zone_timer.start(self.black_zone_time)

    def black_zone_trigger(self):
        self.black_zone_active = False

    def grey_zone_start(self):
        self.grey_zone_active = True
        self.grey_zone_timer.start(self.grey_zone_time)
    
    def trigger_alarm(self,  p):
        if not self.black_zone_active and not self.grey_zone_active:
            self.grey_zone_start()

        elif self.current_p is not None and p > self.current_p and self.black_zone_active:
            self.current_p = p
            self.channel.stop()
            self.alarm_active = True
            self.black_zone_timer.stop()
            self.black_zone_timer.setInterval(self.black_zone_time)
            self.black_zone_timer.start()
            self.black_zone_active = True
            self.channel.play(self.sound[p])
            self.cooldown_active = True
            self.cooldown_timer.start(self.cooldown_time)
            self.alarm_active = False

        elif self.black_zone_active and not self.alarm_active and not self.cooldown_active:
            self.current_p = p
            self.alarm_active = True
            self.black_zone_timer.stop()
            self.black_zone_timer.setInterval(self.black_zone_time)
            self.black_zone_timer.start()            
            self.black_zone_active = True
            self.channel.play(self.sound[p])
            self.cooldown_active = True
            self.cooldown_timer.start(self.cooldown_time)
            self.alarm_active = False

    def end_cooldown(self):
        self.cooldown_active = False

    def alarm_finished(self):
        self.alarm_active = False