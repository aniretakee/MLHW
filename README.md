# MLHW
В ходе ДЗ_1 по МЛ анализировали датасет с данными о машинах.
1. Провели разведочный анализ данных:
	1) нашли пропуски в данных
	2) удалили 1159 объектов с одинаковым признаковым описанием
	3) привели признаки mileage, engine, max_power к вещественному типу
	4) заполнили пропуски в этих столбцах медианными значениями
	5) удалили признак torque
	6) посмотрели на распределение признаков с помощью sns.pairplot() :
		6.1) Явно видна прямая зависимость selling_price от year и обратная зависимость selling_price от km_driven
		6.2) Видно, что коррелируют признаки mileage и engine, mileage и max_power, max_power и engine
		6.3) Проанализировали распределение признаков. Убедились, что выборки train и test похожи между собой. 
	7)  С помощью sns.heatmap() убедились в корреляции признаков  max_power и engine, а также max_power и seats.
	Наименее скоррелированы year и engine.
2. Построили модель LinearRegression() только на числовых признаках. Получили r2_score 0,59 на тренировочной и тестовой выборках.
	1) Стандартизировали вещественные признаки и переобучили модель LinearRegression(). По наибольшему весу определили наиболее информативный признак - year.
Далее работали со стандартизированными признаками:
3. Обучили Lasso регрессию с дефолтными параметрами.
Получили r2_score = 0.58 и почти такие же веса как при LinearRegression(). Это обусловлено тем, что параметр alpha, который по умолчанию = 1, не является оптимальным.
	1) Используя GridSearchCV с 10 фолдами подобрали оптимальный параметр alpha = 26609.
	2) Переобучили Lasso с оптимальным параметром. В результате занулились веса коррелирующих признаков  'mileage', 'engine', а так же 'seats'.
r2_score 0,58 и 0,53 для train и test соответственно.
4. Обучили ElasticNet регрессию, подобрав оптимальные параметры alpha = 1.5 и l1_ratio = 0.9.
r2_score 0,58 и 0,53 для train и test соответственно.

Вывод: лучше всего справилась обычная LinearRegression()

5. Работа с категориальными признаками
	1) Убрали из исходного датасета name и целевую переменную selling_price. 
	2) Признак seats сделали категориальным
	3) С помощью маски cat_features_mask = (df_train.dtypes == "object").values отобрали все категориальные признаки.
	4) Закодировали их с помощью pd.get_dummies()
	5) Получили итоговый датасет всех признаков, соединили вещественные признаки (без seats) с категориальными
6. Обучили Ridge регрессию, подобрав оптимальные параметр alpha = 100.
Получили r2_score 0,63 на train.
На teste проверить модель не получилось.. из-за различий в результате кодирования категориальных признаков для train и test.

Вывод: увеличив количество признаков и используя Ridge регрессию, смогли добиться улучшения качества предсказания.
Ради интереса проверила Ridge регрессию только на вещественных признаках. Получился r2_score 0,59 на train и 0,56 на test.
Т.е основной вклад в улучшение дает именно добавление категориальных признаков в модель.

7. Посчитали долю предиктов, отличающихся от релальных цен не более, чем на 10%.
Получился 21%
Слабенько.

8. Попытка создать сервис fast-api:
	1) Использовали модель LinearRegression на стандартизированных числовых признаках
	2) Подать item на вход не вышло, подаем словарь:)
	В /predict_item  на выходе получаем стоимость ТС
	В /predict_items на выходе список стоимости ТС


