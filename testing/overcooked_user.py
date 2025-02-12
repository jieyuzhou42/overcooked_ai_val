import socket
import pickle
import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

class OvercookedGameClient:
    def __init__(self, server_host="127.0.0.1", server_port=50007):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_host, self.server_port))

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.visualizer = StateVisualizer()

    def send_command(self, action_name, args=[]):
        command = {"action_name": action_name, "args": args}
        self.client_socket.sendall(pickle.dumps(command))

    def receive_state(self):
        data = self.client_socket.recv(4096)
        if not data:
            return None
        return pickle.loads(data)

    def handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                self.client_socket.close()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    self.send_command("up")
                elif event.key == K_DOWN:
                    self.send_command("down")
                elif event.key == K_LEFT:
                    self.send_command("left")
                elif event.key == K_RIGHT:
                    self.send_command("right")

    def render_state(self, game_state):
        surface = self.visualizer.render_state(
            state=game_state["base_env_state"],
            grid=game_state["terrain_mtx"],
            hud_data=StateVisualizer.default_hud_data(game_state["base_env_state"])
        )

        rendered_width, rendered_height = surface.get_size()
        if (rendered_width, rendered_height) != self.screen.get_size():
            self.screen = pygame.display.set_mode((rendered_width, rendered_height), pygame.RESIZABLE)

        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def run(self):
        while True:
            self.handle_pygame_events()

            game_state = self.receive_state()
            if game_state:
                self.render_state(game_state)

            self.clock.tick(60)


if __name__ == "__main__":
    client = OvercookedGameClient()
    client.run()

