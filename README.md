# marky-trader

## Среда
Среда Gymnasium предназначеная для торговли одной валютной парой. Пространство действий представляет собой массив позиций, заданных пользователем. Каждая позиция помечена от -inf до +inf и соответствует соотношению оценки портфеля, задействованного в позиции (> 0, чтобы сделать ставку на рост, < 0, чтобы сделать ставку на снижение). Возможное пространство действий четко задается пользователем

В качестве состояний мы получаем метрики, которые пользователь сам создаст. Их можно создать из таких метрик, как high, low, open, close и value

В качестве реварда агент получает результат функции заданный пользователем. В стандартном случае - % изменения портфеля.

## Запуск
### Зависимости
Все зависимости вы можете увидеть в pyproject.toml
для быстрой установки:
```sh
poetry install
```
### Установка датасета для среды
```sh
python3 downloader.py
```
### Тренировка агента
```sh
python3 <файл с алгоритмом>
```
## Алгоритмы
[Actor Critic](https://github.com/IamVOC/marky-trader/tree/AC)

[DQN](https://github.com/IamVOC/marky-trader/tree/dqn)
## Авторы
Божко Даниил Константинович
Коротун Его Игоревич
Иост Кирилл Витальевич