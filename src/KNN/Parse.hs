{-# OPTIONS_GHC -Wall #-}

module KNN.Parse
  ( runPipeline
  , runWithLoadedModel
  , usage
  ) where

import KNN.Types
import KNN.DataLoader
import KNN.Algorithm
import KNN.Metrics
import KNN.Serialization

-- | Полный конвейер k-NN: загрузка, выбор столбцов, обучение, 
-- предсказание, отчёт.
runPipeline
  :: FilePath -> FilePath -> FilePath -> [String] -> IO ()
runPipeline trainFile testFile modelFile extraArgs = do
  let cfg = parseConfig extraArgs
  mTrain <- loadStep trainFile
  mTest  <- loadStep testFile
  case (mTrain, mTest) of
    (Just trainSet, Just testSet) ->
      applyColumnsStep cfg trainSet testSet modelFile
    _ -> return ()

-- | Загрузить один файл с выборкой; вернуть Nothing при фатальной ошибке.
loadStep :: FilePath -> IO (Maybe TrainingSet)
loadStep path = do
  result <- loadDataset path
  case result of
    Left err -> putStrLn ("Ошибка: " ++ err) >> return Nothing
    Right lr -> do
      mapM_ (\w -> putStrLn ("Предупреждение: " ++ w)) (loadWarnings lr)
      return (Just (loadedPoints lr))

-- | Применить выбор столбцов к обеим выборкам.
applyColumnsStep
  :: Config -> TrainingSet -> TrainingSet -> FilePath -> IO ()
applyColumnsStep cfg trainSet testSet modelFile =
  case configColumns cfg of
    Nothing   -> runWithSets cfg trainSet testSet modelFile
    Just cols ->
      case (selectColumns cols trainSet, selectColumns cols testSet) of
        (Left err, _) -> putStrLn ("Ошибка: " ++ err)
        (_, Left err) -> putStrLn ("Ошибка: " ++ err)
        (Right tr, Right te) -> runWithSets cfg tr te modelFile

-- | Продолжить конвейер после выбора столбцов.
runWithSets
  :: Config -> TrainingSet -> TrainingSet -> FilePath -> IO ()
runWithSets cfg trainSet testSet modelFile =
  case checkDims trainInfo testInfo of
    Left err -> putStrLn ("Ошибка: " ++ err)
    Right () -> runWithDims cfg trainSet testSet modelFile
  where
    trainInfo = datasetInfo trainSet
    testInfo  = datasetInfo testSet

-- | Продолжить конвейер после проверки размерностей.
runWithDims
  :: Config -> TrainingSet -> TrainingSet -> FilePath -> IO ()
runWithDims cfg trainSet testSet modelFile = do
  let trainInfo = datasetInfo trainSet
      testInfo  = datasetInfo testSet
  putStrLn ("Обучающая выборка: "
            ++ show (infoNumPoints trainInfo) ++ " объектов, "
            ++ show (infoNumFeatures trainInfo) ++ " признаков")
  putStrLn ("Тестовая выборка : "
            ++ show (infoNumPoints testInfo) ++ " объектов")
  putStrLn ("Конфигурация     : " ++ show cfg)
  case trainModel cfg trainSet of
    Left err    -> putStrLn ("Ошибка: " ++ err)
    Right model -> runWithModel model testSet modelFile

-- | Продолжить конвейер после обучения модели.
runWithModel :: Model -> TrainingSet -> FilePath -> IO ()
runWithModel model testSet modelFile = do
  saveResult <- saveModel modelFile model
  case saveResult of
    Left err -> putStrLn (
        "Предупреждение: не удалось сохранить модель: " ++ err
        )
    Right _  -> putStrLn ("Модель сохранена в файл: " ++ modelFile)
  let queries = map (\dp -> (dpFeatures dp, dpLabel dp)) testSet
      preds   = predictAll model queries
  putStrLn "\n=== Предсказания ==="
  mapM_ printPrediction preds
  putStr (formatReport preds)

runWithLoadedModel :: FilePath -> FilePath -> IO ()
runWithLoadedModel modelFile testFile = do
  loadResult <- loadModel modelFile
  case loadResult of
    Left err -> do
      putStrLn ("Ошибка загрузки модели: " ++ err)
      putStrLn "Попробуйте обучить модель заново."
    Right model -> do
      mTest <- loadStep testFile
      case mTest of
        Just testSet -> do
          putStrLn ("Модель загружена из файла: " ++ modelFile)
          
          let cfg = modelConfig model
              trainSet = modelTrainingSet model
              trainInfo = datasetInfo trainSet
              testInfo = datasetInfo testSet
          
          putStrLn (
            "Модель обучена на: " 
            ++ show (infoNumPoints trainInfo) 
            ++ " объектов, "
            ++ show (infoNumFeatures trainInfo) 
            ++ " признаков"
            )
          putStrLn ("Конфигурация модели: k=" ++ show (configK cfg) 
                    ++ ", метрика=" ++ show (configMetric cfg)
                    ++ ", задача=" ++ show (configTaskType cfg))
          putStrLn (
            "Тестовая выборка : " 
            ++ show (infoNumPoints testInfo) 
            ++ " объектов, "
            ++ show (infoNumFeatures testInfo) 
            ++ " признаков"
            )
          
          if infoNumFeatures testInfo /= infoNumFeatures trainInfo
            then putStrLn $ "Ошибка: тестовая выборка имеет " 
                            ++ show (infoNumFeatures testInfo) 
                            ++ " признаков, а модель ожидает " 
                            ++ show (infoNumFeatures trainInfo)
            else do
              let queries = map (\dp -> (dpFeatures dp, dpLabel dp)) testSet
                  preds   = predictAll model queries
              putStrLn "\n=== Предсказания ==="
              mapM_ printPrediction preds
              putStr (formatReport preds)
        Nothing -> return ()

-- | Вывести одну строку с результатом предсказания.
printPrediction :: PredictionResult -> IO ()
printPrediction pr =
  putStrLn
    ( "предсказано=" ++ show (prPredicted pr)
      ++ "  истинное=" ++ show (prActual pr) )

-- | Разобрать необязательные аргументы в структуру Config.
-- По умолчанию: k=3, метрика=Euclidean, задача=Classification, все столбцы.
parseConfig :: [String] -> Config
parseConfig args =
  Config
    { configK        = k
    , configMetric   = metric
    , configTaskType = taskType
    , configColumns  = cols
    }
  where
    k = case args of
          (kStr : _) ->
            case reads kStr of
              [(n, "")] | n > 0 -> n
              _                  -> defaultK
          [] -> defaultK
    metric = case args of
               (_ : s : _) -> parseMetric s
               _           -> Euclidean
    taskType = case args of
                 (_ : _ : s : _) -> parseTaskType s
                 _               -> Classification
    cols = case args of
             (_ : _ : _ : s : _) -> parseColumns s
             _                   -> Nothing
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

-- | Разобрать список столбцов вида "0,1,3" в [Int].
-- Возвращает Nothing если строка не является списком неотрицательных чисел.
parseColumns :: String -> Maybe [Int]
parseColumns s =
  let parts = splitOnComma s
  in mapM readNonNeg parts
  where
    splitOnComma str =
      case break (== ',') str of
        (tok, [])       -> [tok]
        (tok, _ : rest) -> tok : splitOnComma rest
    readNonNeg str =
      case reads str of
        [(n, "")] | n >= 0 -> Just n
        _                  -> Nothing

-- | Строка справки по использованию программы.
usage :: String
usage = unlines
  [ "Использование: stack run -- <обучение> <тест> <модель>"
  , "              [k] [метрика] [задача] [столбцы]"
  , "  обучение — путь к файлу с обучающей выборкой"
  , "  тест     — путь к файлу с тестовой выборкой"
  , "  модель   — путь для сохранения/загрузки обученной модели"
  , "  k        — количество соседей (по умолчанию: 3)"
  , "  метрика  — Euclidean | Manhattan | Chebyshev"
  , "             (по умолчанию: Euclidean)"
  , "  задача   — Classification | Regression"
  , "             (по умолчанию: Classification)"
  , "  столбцы  — индексы признаков через запятую, например 0,1,3"
  , "             (по умолчанию: все столбцы; последний — метка)"
  , ""
  , "Формат данных: числа через запятую или пробел."
  ]