# MLOps

Цель проекта - создание алгоритма для детекции мошеннических операций с картами. 
Основные проблемы - сильный дисбаланс классов и высокая "стоимость" как ложно отрицательного, так и ложно положительного ответа.

С учетом данных проблем были выбраны следующие метрики:
- AUC ROC - стандартная кривая отношения отклика и процента ложноположительных ответов
- AP - средняя точность, для выделения решений с высокой точностью положительных ответов
- CP@k - точность первых k карт, для повышения точности в рамках ежедневного взаимодействия оператора с системой
