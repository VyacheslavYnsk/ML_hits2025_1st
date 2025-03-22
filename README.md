# ML_hits2025_1st

### Янущик Вячеслав Дмитриевич, 972302 (2)

Работал в macOS

### Обучить модель 

 #### python model.py train --dataset=data/tsumladvanced2025/train.csv

### Сделать прогноз

 #### python model.py predict --dataset=data/tsumladvanced2025/test.csv


## ДОКЕР : 

 #### docker build -t ml_hits2025

### Прогнозы:

#### docker run -v $(pwd)/data:/data ml_hits2025 poetry run python model.py predict --dataset=/data/tsumladvanced2025/test.csv

### Обучение: 

#### docker run -v $(pwd)/data:/data ml_hits2025 poetry run python model.py train --dataset=/data/tsumladvanced2025/train.csv



## Дополнительно

После прогноза результат загружается в results.csv,
модель сохраняется в файл model.pkl
В освоении материала использовал курсы https://habr.com/ru/companies/ods/articles/322626/ , мог воспользоваться GPT для нахожения ошибок и недочетов. 

Работал с clearml ЧЕРЕЗ jupyter! (он есть в папке), загрузил на него все данные, также optuna использовал только в jupyter. Снимки clearml в /data/screens

## РАБОТА С ДАННЫМИ:
 ### Пустные параметры пассажира заполнял согласно видео
 ### Булевые переменные перевожу в числовые, добавляю к пассажиру параметры 

 ### Сначала работал в jupyter (он лежит в репе /notebooks). В jupyter использовал optuna для подбора гиперпараметров. Изначально я выбрал три модели: LR, XGB и RF. RF выдал худшие результаты, поэтому его в My_Classifier_Model нет. 