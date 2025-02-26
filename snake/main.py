import pygame
import random
import sys
import math


# 初始化Pygame
pygame.init()
# 初始化混音器
pygame.mixer.init()

# 定义颜色
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)

# 设置游戏窗口大小
width, height = 1200, 1200
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# 定义游戏参数
cell_size = 20
speed = 10
food_score = 10
power_up_score = 15
bullet_cooldown = 2000  # 子弹冷却时间（毫秒）
enemy_interval = 20000  # 敌人生成间隔（毫秒）
enemy_chase_interval = 20000  # 敌人追逐间隔（毫秒）
eat_distance = cell_size * 1.5  # 吃到食物的距离阈值
attack_distance = cell_size * 2  # 攻击距离阈值
sight_distance = cell_size * 20  # 视线距离阈值

# 定义状态
PATROL = "patrol"
CHASE = "chase"
ATTACK = "attack"

# 加载背景音乐
pygame.mixer.music.load('background_music.mp3')
# 设置音乐循环播放
pygame.mixer.music.play(-1)
# 设置背景音乐音量
pygame.mixer.music.set_volume(0.5)

# 加载游戏结束音效
game_over_sound = pygame.mixer.Sound('game_over_sound.wav')
# 加载敌人在周围音效
enemy_near_sound = pygame.mixer.Sound('enemy_near_sound.wav')


# 行为树节点基类
class TreeNode:
    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def evaluate(self, snake, food, power_up, enemies):
        pass


# 条件节点
class ConditionNode(TreeNode):
    def __init__(self, condition):
        super().__init__()
        self.condition = condition

    def evaluate(self, snake, food, power_up, enemies):
        return self.condition(snake, food, power_up, enemies)


# 动作节点
class ActionNode(TreeNode):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def evaluate(self, snake, food, power_up, enemies):
        self.action(snake, food, power_up, enemies)


# 选择节点（只要有一个子节点返回True，就返回True）
class SelectorNode(TreeNode):
    def evaluate(self, snake, food, power_up, enemies):
        for child in self.children:
            if child.evaluate(snake, food, power_up, enemies):
                return True
        return False


# 序列节点（所有子节点都返回True，才返回True）
class SequenceNode(TreeNode):
    def evaluate(self, snake, food, power_up, enemies):
        for child in self.children:
            if not child.evaluate(snake, food, power_up, enemies):
                return False
        return True


# 定义蛇类
class Snake:
    def __init__(self):
        self.body = [(width // 2, height // 2)]
        self.direction = (0, -1)
        self.score = 0
        self.lives = 3
        self.is_invincible = False
        self.is_pass_wall = False
        self.is_double_score = False
        self.is_bullet_active = False
        self.bullet_position = (0, 0)
        self.last_turn_time = pygame.time.get_ticks()
        self.bullet_speed = cell_size * 2  # 子弹速度为敌人速度的2倍
        self.bullet_direction = (0, 0)  # 新增属性，记录子弹发射方向

    def move(self):
        new_head = (self.body[0][0] + self.direction[0] * cell_size,
                    self.body[0][1] + self.direction[1] * cell_size)
        self.body.insert(0, new_head)
        if self.is_close_to(food.position, eat_distance):
            self.score += food_score
            if self.is_double_score:
                self.score *= 2
            generate_food()
        elif self.is_close_to(power_up.position, eat_distance):
            self.score += power_up_score
            self.apply_power_up()
            generate_power_up()
        else:
            self.body.pop()

    def check_collision(self):
        if not self.is_pass_wall:
            if self.body[0][0] < 0 or self.body[0][0] >= width or \
                    self.body[0][1] < 0 or self.body[0][1] >= height:
                self.lives = 0
            for part in self.body[1:]:
                if self.body[0] == part:
                    self.lives = 0
        else:
            if self.body[0][0] < 0:
                self.body[0] = (width - cell_size, self.body[0][1])
            elif self.body[0][0] >= width:
                self.body[0] = (0, self.body[0][1])
            elif self.body[0][1] < 0:
                self.body[0] = (self.body[0][0], height - cell_size)
            elif self.body[0][1] >= height:
                self.body[0] = (self.body[0][0], 0)

    def apply_power_up(self):
        if power_up.type == 'invincible':
            self.is_invincible = True
            pygame.time.set_timer(pygame.USEREVENT + 1, 15000)
        elif power_up.type == 'pass_wall':
            self.is_pass_wall = True
            pygame.time.set_timer(pygame.USEREVENT + 2, 15000)
        elif power_up.type == 'double_score':
            self.is_double_score = True
            pygame.time.set_timer(pygame.USEREVENT + 3, 15000)

    def is_close_to(self, target, distance):
        x1, y1 = self.body[0]
        x2, y2 = target
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= distance


# 定义食物类
class Food:
    def __init__(self):
        self.position = (0, 0)
        self.type = None

    def generate(self):
        while True:
            x = random.randint(0, (width // cell_size) - 1) * cell_size
            y = random.randint(0, (height // cell_size) - 1) * cell_size
            if (x, y) not in snake.body and (x, y)!= power_up.position:
                self.position = (x, y)
                self.type = random.choice(['normal','speed','slow', 'life'])
                break


# 定义道具类
class PowerUp:
    def __init__(self):
        self.position = (0, 0)
        self.type = None

    def generate(self):
        while True:
            x = random.randint(0, (width // cell_size) - 1) * cell_size
            y = random.randint(0, (height // cell_size) - 1) * cell_size
            if (x, y) not in snake.body and (x, y)!= food.position:
                self.position = (x, y)
                self.type = random.choice(['invincible', 'pass_wall', 'double_score'])
                break


# 定义敌人类
class Enemy:
    def __init__(self):
        self.position = (0, 0)
        self.direction = (1, 0)
        self.last_turn_time = pygame.time.get_ticks()
        self.state = PATROL
        self.patrol_points = []
        self.current_patrol_index = 0
        self.generate_patrol_points()
        self.smooth_path = []
        self.smooth_path_index = 0
        self.base_speed = cell_size
        self.update_speed()
        self.has_started_moving = False

    def generate_patrol_points(self):
        num_points = random.randint(3, 5)
        for _ in range(num_points):
            x = random.randint(0, (width // cell_size) - 1) * cell_size
            y = random.randint(0, (height // cell_size) - 1) * cell_size
            self.patrol_points.append((x, y))

    def update_speed(self):
        if self.state == PATROL:
            self.speed = self.base_speed * 1.5
        elif self.state in [CHASE, ATTACK]:
            self.speed = self.base_speed * 0.5

    def move(self):
        if not self.has_started_moving:
            self.has_started_moving = True
            # 这里可以添加一些初始化移动逻辑，比如移动几步后再开始正常状态
            self.move()
            return
        self.update_speed()
        if self.state == CHASE:
            dx = snake.body[0][0] - self.position[0]
            dy = snake.body[0][1] - self.position[1]
            if dx!= 0 or dy!= 0:
                length = math.sqrt(dx ** 2 + dy ** 2)
                self.direction = (dx / length, dy / length)
            new_x = self.position[0] + self.direction[0] * self.speed
            new_y = self.position[1] + self.direction[1] * self.speed
            if 0 <= new_x < width and 0 <= new_y < height:
                self.position = (new_x, new_y)
        elif self.state == PATROL:
            if not self.smooth_path or self.smooth_path_index >= len(self.smooth_path):
                target = self.patrol_points[self.current_patrol_index]
                self.smooth_path = self.generate_smooth_path(self.position, target)
                self.smooth_path_index = 0
            new_x, new_y = self.smooth_path[self.smooth_path_index]
            if 0 <= new_x < width and 0 <= new_y < height:
                self.position = (new_x, new_y)
                self.smooth_path_index += 1
            if self.position == self.patrol_points[self.current_patrol_index]:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
        elif self.state == ATTACK:
            pass

    def generate_smooth_path(self, start, end):
        num_points = 5  # 插入的中间点数
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            points.append((x, y))
        return points

    def can_see_player(self):
        dx = snake.body[0][0] - self.position[0]
        dy = snake.body[0][1] - self.position[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance > sight_distance:
            return False
        angle_to_player = math.atan2(dy, dx)
        angle_difference = abs(angle_to_player - math.atan2(self.direction[1], self.direction[0]))
        return angle_difference < math.pi / 4

    def change_state(self, new_state):
        self.state = new_state
        self.update_speed()


# 初始化游戏对象
snake = Snake()
food = Food()
power_up = PowerUp()
enemies = []
last_bullet_time = pygame.time.get_ticks()
last_enemy_time = pygame.time.get_ticks()
last_enemy_chase_time = pygame.time.get_ticks()
next_enemy_time = last_enemy_time + enemy_interval


# 生成食物和道具
def generate_food():
    food.generate()


def generate_power_up():
    power_up.generate()


# 绘制游戏界面
def draw_game():
    global next_enemy_time
    screen.fill(BLACK)
    # 绘制蛇
    for part in snake.body:
        pygame.draw.rect(screen, GREEN, (part[0], part[1], cell_size, cell_size))
    # 绘制食物
    if food.type == 'normal':
        pygame.draw.rect(screen, WHITE, (food.position[0], food.position[1], cell_size, cell_size))
    elif food.type =='speed':
        pygame.draw.rect(screen, GREEN, (food.position[0], food.position[1], cell_size, cell_size))
    elif food.type =='slow':
        pygame.draw.rect(screen, RED, (food.position[0], food.position[1], cell_size, cell_size))
    elif food.type == 'life':
        pygame.draw.rect(screen, BLUE, (food.position[0], food.position[1], cell_size, cell_size))
    # 绘制道具
    if power_up.type:
        draw_star(screen, power_up.position, cell_size, get_color_by_power_up_type(power_up.type))
    # 绘制敌人
    for enemy in enemies:
        draw_arrow(screen, enemy.position, cell_size, RED)
    # 绘制子弹
    if snake.is_bullet_active:
        draw_triangle(screen, snake.bullet_position, cell_size, RED)
    # 显示得分和生命数
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {snake.score}", True, WHITE)
    lives_text = font.render(f"Lives: {snake.lives}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (10, 50))

    # 计算并显示距离下一个敌人来临的时间
    remaining_time = (next_enemy_time - pygame.time.get_ticks()) // 1000
    if remaining_time > 0:
        enemy_time_text = font.render(f"Next Enemy: {remaining_time}s", True, WHITE)
        screen.blit(enemy_time_text, (10, 90))

    pygame.display.flip()


def draw_star(screen, position, size, color):
    points = []
    for i in range(5):
        angle = i * (2 * math.pi / 5) + math.pi / 2
        x = position[0] + size / 2 * math.cos(angle)
        y = position[1] + size / 2 * math.sin(angle)
        points.append((x, y))
        angle = (i * (2 * math.pi / 5) + math.pi / 2) + math.pi / 5
        x = position[0] + size / 4 * math.cos(angle)
        y = position[1] + size / 4 * math.sin(angle)
        points.append((x, y))
    pygame.draw.polygon(screen, color, points)


def draw_arrow(screen, position, size, color):
    points = [(position[0], position[1]),
              (position[0] + size, position[1] + size // 2),
              (position[0], position[1] + size),
              (position[0] + size // 2, position[1] + size // 2)]
    pygame.draw.polygon(screen, color, points)


def draw_triangle(screen, position, size, color):
    points = [(position[0], position[1]),
              (position[0] + size, position[1]),
              (position[0] + size // 2, position[1] + size)]
    pygame.draw.polygon(screen, color, points)


def get_color_by_power_up_type(power_up_type):
    if power_up.type == 'invincible':
        return YELLOW
    elif power_up.type == 'pass_wall':
        return CYAN
    elif power_up.type == 'double_score':
        return MAGENTA


# 检查碰撞
def check_collision():
    global enemy_near_sound_playing
    enemy_near_sound_playing = False
    for enemy in enemies:
        for part in snake.body:
            if part == enemy.position:
                if not snake.is_invincible:
                    snake.lives -= 1
        if snake.is_bullet_active:
            bullet_x_diff = abs(snake.bullet_position[0] - enemy.position[0])
            bullet_y_diff = abs(snake.bullet_position[1] - enemy.position[1])
            if bullet_x_diff < cell_size and bullet_y_diff < cell_size:
                enemies.remove(enemy)
                snake.is_bullet_active = False
        distance = math.sqrt((snake.body[0][0] - enemy.position[0]) ** 2 + (snake.body[0][1] - enemy.position[1]) ** 2)
        if distance <= attack_distance:
            enemy.change_state(ATTACK)
            if not snake.is_invincible:
                snake.lives -= 1
            if not enemy_near_sound_playing:
                enemy_near_sound.play()
                enemy_near_sound_playing = True


# 显示游戏说明
def show_instructions():
    font = pygame.font.Font(None, 36)
    instructions = [
        "游戏说明:",
        "1. 控制: 方向键控制移动",
        "2. 食物:",
        "   - 白色: 普通食物 (+10分)",
        "   - 绿色: 加速食物 (+20分, 加速)",
        "   - 红色: 减速食物 (+5分, 减速)",
        "   - 蓝色: 生命食物 (+15分, +1生命)",
        "3. 道具 (五角星):",
        "   - 金色: 无敌 (15秒)",
        "   - 青色: 穿墙 (15秒)",
        "   - 紫色: 双倍分数 (15秒)",
        "4. 敌人:",
        "   - 红色弓箭形状",
        "   - 每20秒生成一个",
        "   - 按q发射子弹消灭敌人 (-30分)",
        "5. 游戏结束:",
        "   - 生命耗尽",
        "   - 被敌人抓住",
        "按任意键开始游戏..."
    ]
    for i, line in enumerate(instructions):
        text = font.render(line, True, WHITE)
        screen.blit(text, (50, 50 + i * 40))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                waiting = False


# 行为树相关条件和动作
def is_close_to_food(snake, food, power_up, enemies):
    return snake.is_close_to(food.position, eat_distance * 2)


def move_towards_food(snake, food, power_up, enemies):
    x_diff = food.position[0] - snake.body[0][0]
    y_diff = food.position[1] - snake.body[0][1]
    if abs(x_diff) > abs(y_diff):
        if x_diff > 0:
            snake.direction = (1, 0)
        else:
            snake.direction = (-1, 0)
    else:
        if y_diff > 0:
            snake.direction = (0, 1)
        else:
            snake.direction = (0, -1)


def is_close_to_enemy(snake, food, power_up, enemies):
    for enemy in enemies:
        if snake.is_close_to(enemy.position, eat_distance * 2):
            return True
    return False


def move_away_from_enemy(snake, food, power_up, enemies):
    for enemy in enemies:
        if snake.is_close_to(enemy.position, eat_distance * 2):
            x_diff = enemy.position[0] - snake.body[0][0]
            y_diff = enemy.position[1] - snake.body[0][1]
            if abs(x_diff) > abs(y_diff):
                if x_diff > 0:
                    snake.direction = (-1, 0)
                else:
                    snake.direction = (1, 0)
            else:
                if y_diff > 0:
                    snake.direction = (0, -1)
                else:
                    snake.direction = (0, 1)


# 构建行为树
root = SelectorNode()
food_sequence = SequenceNode()
food_sequence.add_child(ConditionNode(is_close_to_food))
food_sequence.add_child(ActionNode(move_towards_food))
root.add_child(food_sequence)

enemy_sequence = SequenceNode()
enemy_sequence.add_child(ConditionNode(is_close_to_enemy))
enemy_sequence.add_child(ActionNode(move_away_from_enemy))
root.add_child(enemy_sequence)


# 主函数
def main():
    global last_bullet_time, last_enemy_time, last_enemy_chase_time, next_enemy_time, enemy_near_sound_playing
    clock = pygame.time.Clock()
    running = True
    generate_food()
    generate_power_up()
    next_enemy_time = last_enemy_time + enemy_interval
    enemy_near_sound_playing = False
    show_instructions()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction[1] == 0:
                    snake.direction = (0, -1)
                    snake.last_turn_time = pygame.time.get_ticks()
                elif event.key == pygame.K_DOWN and snake.direction[1] == 0:
                    snake.direction = (0, 1)
                    snake.last_turn_time = pygame.time.get_ticks()
                elif event.key == pygame.K_LEFT and snake.direction[0] == 0:
                    snake.direction = (-1, 0)
                    snake.last_turn_time = pygame.time.get_ticks()
                elif event.key == pygame.K_RIGHT and snake.direction[0] == 0:
                    snake.direction = (1, 0)
                    snake.last_turn_time = pygame.time.get_ticks()
                elif event.key == pygame.K_q:
                    current_time = pygame.time.get_ticks()
                    if current_time - last_bullet_time >= bullet_cooldown and snake.score >= 30:
                        snake.is_bullet_active = True
                        snake.bullet_position = snake.body[0]
                        snake.bullet_direction = snake.direction
                        last_bullet_time = current_time
                        snake.score -= 30
            elif event.type == pygame.USEREVENT + 1:
                snake.is_invincible = False
            elif event.type == pygame.USEREVENT + 2:
                snake.is_pass_wall = False
            elif event.type == pygame.USEREVENT + 3:
                snake.is_double_score = False

        # 行为树控制蛇的移动
        root.evaluate(snake, food, power_up, enemies)

        snake.move()
        snake.check_collision()

        # 更新子弹位置
        if snake.is_bullet_active:
            bullet_dx = snake.bullet_direction[0] * snake.bullet_speed
            bullet_dy = snake.bullet_direction[1] * snake.bullet_speed
            new_bullet_x = snake.bullet_position[0] + bullet_dx
            new_bullet_y = snake.bullet_position[1] + bullet_dy
            # 检查子弹是否超出边界
            if 0 <= new_bullet_x < width and 0 <= new_bullet_y < height:
                snake.bullet_position = (new_bullet_x, new_bullet_y)
                for enemy in enemies[:]:
                    if (abs(snake.bullet_position[0] - enemy.position[0]) < cell_size and
                            abs(snake.bullet_position[1] - enemy.position[1]) < cell_size):
                        enemies.remove(enemy)
                        snake.is_bullet_active = False
            else:
                snake.is_bullet_active = False

        for enemy in enemies:
            enemy.move()
            if enemy.can_see_player():
                enemy.change_state(CHASE)
            else:
                enemy.change_state(PATROL)
            distance = math.sqrt((snake.body[0][0] - enemy.position[0]) ** 2 +
                                 (snake.body[0][1] - enemy.position[1]) ** 2)
            if distance <= attack_distance:
                enemy.change_state(ATTACK)
        check_collision()

        current_time = pygame.time.get_ticks()
        if current_time >= next_enemy_time:
            new_enemy = Enemy()
            enemies.append(new_enemy)
            last_enemy_time = current_time
            next_enemy_time = current_time + enemy_interval

        if current_time - last_enemy_chase_time >= enemy_chase_interval:
            for enemy in enemies:
                enemy.move()
            last_enemy_chase_time = current_time

        draw_game()

        if snake.lives <= 0:
            running = False
            game_over_sound.play()

        clock.tick(speed)

    # 游戏结束
    font = pygame.font.Font(None, 72)
    game_over_text = font.render("Game Over", True, WHITE)
    screen.blit(game_over_text, (width // 2 - 150, height // 2 - 50))
    pygame.display.flip()
    pygame.time.wait(2000)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()