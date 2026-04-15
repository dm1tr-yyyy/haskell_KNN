{-# OPTIONS_GHC -Wall #-}

-- Точка входа: разбор аргументов командной строки, запуск
-- k-NN и вывод читаемого отчёта.
--
-- Использование:
--   stack run -- <обучающий-файл> <тестовый-файл> <файл-модели> [k] [метрика] [задача]
--
-- Допустимые значения метрики: Euclidean | Manhattan | Chebyshev
-- Допустимые значения задачи : Classification | Regression
-- По умолчанию: k=3, метрика=Euclidean, задача=Classification
module Main (main) where

import System.Environment (getArgs)
import System.Exit        (exitFailure)

import Lib

-- Точка входа в программу.
main :: IO ()
main = do
  args <- getArgs
  case args of
    (trainFile : testFile : modelFile : rest) ->
      runPipeline trainFile testFile modelFile rest
    _ -> do
      putStrLn usage
      exitFailure

-- Полный pipeline k-NN: загрузка данных, обучение, предсказание,
-- оценка качества и сохранение модели.
runPipeline
  :: FilePath  -- Файл с обучающей выборкой
  -> FilePath  -- Файл с тестовой выборкой (или данными для предсказания)
  -> FilePath  -- Файл для сохранения/загрузки модели
  -> [String]  -- Необязательные аргументы: [k] [метрика] [задача]
  -> IO ()
runPipeline trainFile testFile modelFile extraArgs = do
  let cfg = parseConfig extraArgs

  -- Загрузка обучающей выборки
  trainResult <- loadDataset trainFile
  trainSet <- exitOnError trainResult

  -- Загрузка тестовой выборки
  testResult <- loadDataset testFile
  testSet <- exitOnError testResult

  -- Информация о данных
  let info = datasetInfo trainSet
  putStrLn ("Обучающая выборка: "
            ++ show (infoNumPoints info) ++ " объектов, "
            ++ show (infoNumFeatures info) ++ " признаков")
  putStrLn ("Тестовая выборка : "
            ++ show (length testSet) ++ " объектов")
  putStrLn ("Конфигурация     : " ++ show cfg)

  -- Обучение модели
  model <- exitOnError (trainModel cfg trainSet)

  -- Сохранение модели
  saveResult <- saveModel modelFile model
  case saveResult of
    Left err ->
      putStrLn ("Предупреждение: не удалось сохранить модель: " ++ err)
    Right _  ->
      putStrLn ("Модель сохранена в файл: " ++ modelFile)

  -- Предсказание на тестовой выборке
  let queries  =
        map (\dp -> (dpFeatures dp, Just (dpLabel dp))) testSet
      rawPreds = predictAll model queries
  preds <- mapM exitOnError rawPreds

  -- Вывод предсказаний
  putStrLn "\n=== Предсказания ==="
  mapM_ printPrediction preds

  -- Оценка качества
  putStr (formatReport preds)

-- | Вывести одну строку с результатом предсказания.
printPrediction :: PredictionResult -> IO ()
printPrediction pr =
  putStrLn
    ( "предсказано=" ++ show (prPredicted pr)
      ++ case prActual pr of
           Nothing -> ""
           Just a  -> "  истинное=" ++ show a )

-- | Разобрать необязательные аргументы в структуру Config.
-- По умолчанию: k=3, метрика=Euclidean, задача=Classification.
parseConfig :: [String] -> Config
parseConfig args =
  Config
    { configK        = k
    , configMetric   = metric
    , configTaskType = taskType
    }
  where
    k = case args of
          (kStr : _) ->
            case reads kStr of
              [(n, "")] | n > 0 -> n
              _                  -> defaultK
          [] -> defaultK
    metric = case args of
               (_ : metricStr : _) -> parseMetric metricStr
               _                   -> Euclidean
    taskType = case args of
                 (_ : _ : taskStr : _) -> parseTaskType taskStr
                 _                     -> Classification
    defaultK = 3

-- | Разобрать название метрики из строки.
parseMetric :: String -> DistanceMetric
parseMetric s =
  case s of
    "Euclidean" -> Euclidean
    "Manhattan" -> Manhattan
    "Chebyshev" -> Chebyshev
    _           -> Euclidean

-- | Разобрать тип задачи из строки.
parseTaskType :: String -> TaskType
parseTaskType s =
  case s of
    "Regression" -> Regression
    _            -> Classification

-- | Вывести сообщение об ошибке и завершить программу при Left;
-- вернуть значение при Right.
exitOnError :: Either String a -> IO a
exitOnError (Right v)  = return v
exitOnError (Left err) = do
  putStrLn ("Ошибка: " ++ err)
  exitFailure

-- | Строка справки по использованию программы.
usage :: String
usage = unlines
  [ "Использование: stack run -- <обучение> <тест> <модель> [k] [метрика] [задача]"
  , "  обучение — путь к файлу с обучающей выборкой"
  , "  тест     — путь к файлу с тестовой выборкой"
  , "  модель   — путь для сохранения/загрузки обученной модели"
  , "  k        — количество соседей (по умолчанию: 3)"
  , "  метрика  — Euclidean | Manhattan | Chebyshev (по умолчанию: Euclidean)"
  , "  задача   — Classification | Regression (по умолчанию: Classification)"
  , ""
  , "Формат файла данных: числа через запятую или пробел, последний столбец — метка."
  ]
