{-# OPTIONS_GHC -Wall #-}

-- | Основные типы данных алгоритма кнн
module KNN.Types
  ( DistanceMetric (..)
  , TaskType (..)
  , Label
  , Features
  , DataPoint (..)
  , TrainingSet
  , Config (..)
  , Model (..)
  , PredictionResult (..)
  ) where

data DistanceMetric
  = Euclidean
  -- ^ Евклидово расстояние L2: sqrt(сумма квадратов разностей)
  | Manhattan
  -- ^ Манхэттенское расстояние L1: сумма модулей разностей
  | Chebyshev
  -- ^ Расстояние Чебышёва L-inf: максимум модулей разностей
  deriving (Show, Read, Eq)

-- | Тип решаемой задачи.
data TaskType
  = Classification
  -- ^ Классификация: предсказание дискретной метки голосованием большинства
  | Regression
  -- ^ Регрессия: предсказание непрерывного значения как среднего соседей
  deriving (Show, Read, Eq)

-- | Метка класса или целевое значение регрессии (вещественное число).
type Label = Double

-- | Вектор признаков одного объекта.
type Features = [Double]

-- | Один размеченный объект обучающей выборки.
data DataPoint = DataPoint
  { dpFeatures :: Features
  -- ^ Вектор признаков
  , dpLabel    :: Label
  -- ^ Метка класса или целевое значение
  } deriving (Show, Read, Eq)

-- | Обучающая выборка — список размеченных объектов.
type TrainingSet = [DataPoint]

-- | Гиперпараметры и настройки модели.
data Config = Config
  { configK        :: Int
  -- ^ Количество ближайших соседей
  , configMetric   :: DistanceMetric
  -- ^ Используемая метрика расстояния
  , configTaskType :: TaskType
  -- ^ Тип задачи: классификация или регрессия
  } deriving (Show, Read, Eq)

-- | Обученная модель: настройки + сохранённая обучающая выборка.
data Model = Model
  { modelConfig      :: Config
  -- ^ Гиперпараметры модели
  , modelTrainingSet :: TrainingSet
  -- ^ Обучающая выборка
  } deriving (Show, Read, Eq)

-- | Результат предсказания для одного объекта.
data PredictionResult = PredictionResult
  { prFeatures  :: Features
  -- ^ Вектор признаков классифицированного объекта
  , prPredicted :: Label
  -- ^ Метка, предсказанная моделью
  , prActual    :: Maybe Label
  -- ^ Истинная метка, если известна
  } deriving (Show, Eq)
