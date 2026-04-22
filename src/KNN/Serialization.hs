{-# OPTIONS_GHC -Wall #-}

-- | Сохранение и загрузка обученной модели k-NN в текстовый файл
-- с помощью экземпляров Show/Read.
module KNN.Serialization
  ( saveModel
  , loadModel
  ) where

import KNN.Types (Model)

-- | Сериализовать модель в текстовый файл.
saveModel :: FilePath -> Model -> IO (Either String ())
saveModel path model = do
  writeFile path (show model)
  return (Right ())

-- | Десериализовать модель из файла, ранее записанного функцией 'saveModel'.
loadModel :: FilePath -> IO (Either String Model)
loadModel path = do
  content <- readFile path
  return $ case reads content of
    [(model, _)] -> Right model
    _            -> Left ("Не удалось разобрать файл модели: " ++ show path)
