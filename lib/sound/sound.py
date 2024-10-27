import pygame
import time


def play_mp3(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play(1)


def start_mp3():
    play_mp3('lib/sound/start.mp3')
    time.sleep(1.1)

def end_mp3():
    play_mp3('lib/sound/run.mp3')
    time.sleep(1)