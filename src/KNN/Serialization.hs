{-# OPTIONS_GHC -Wall #-}

-- | Сохранение и загрузка обученной модели k-NN в текстовый файл
-- с помощью экземпляров Show/Read.
module KNN.Serialization
  ( saveModel
  , loadModel
  ) where

import Control.Exception (SomeException, try)

import KNN.Types (Model)

-- | Сериализовать модель в текстовый файл.
-- Возвращает Left с сообщением об ошибке при неудаче.
saveModel :: FilePath -> Model -> IO (Either String ())
saveModel path model = do
  result <-
    try (writeFile path (show model)) :: IO (Either SomeException ())
  return $ case result of
    Left ex -> Left ("Не удалось сохранить модель: " ++ show ex)
    Right _ -> Right ()

-- | Десериализовать модель из файла, ранее записанного функцией 'saveModel'.
-- Возвращает Left с сообщением об ошибке при неудаче.
loadModel :: FilePath -> IO (Either String Model)
loadModel path = do
  result <-
    try (readFile path) :: IO (Either SomeException String)
  return $ case result of
    Left ex       -> Left ("Не удалось прочитать файл модели: " ++ show ex)
    Right content ->
      case reads content of
        [(model, _)] -> Right model
        _            ->
          Left ("Не удалось разобрать файл модели: " ++ show path)
