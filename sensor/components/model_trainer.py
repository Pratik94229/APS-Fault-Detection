from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import os,sys 
from xgboost import XGBClassifier
from sensor import utils
from sklearn.metrics import f1_score


class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    def fine_tune(self):
        try:
            logging.info("Performing hyperparameter tuning using RandomizedSearchCV...")
            
            # Define the hyperparameter grid to search
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9],
                'n_estimators': [50, 100, 150, 200],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }

            # Create the XGBoost classifier
            xgb_clf = XGBClassifier()

            # Create the RandomizedSearchCV object
            randomized_search = RandomizedSearchCV(
                xgb_clf,
                param_distributions=param_grid,
                n_iter=10,  # Number of iterations for random search
                scoring='f1',  # Use F1 score as the metric for evaluation
                n_jobs=-1,  # Use all available CPU cores for parallel processing
                cv=StratifiedKFold(n_splits=5, shuffle=True),  # Cross-validation strategy
                verbose=3,
                random_state=42
            )

            # Load the training data
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]

            # Perform the hyperparameter search
            randomized_search.fit(x_train, y_train)

            # Get the best estimator and best hyperparameters
            best_model = randomized_search.best_estimator_
            best_params = randomized_search.best_params_

            logging.info("Hyperparameter tuning complete. Best hyperparameters:")
            logging.info(best_params)

            return best_model

        except Exception as e:
            raise SensorException(e, sys)

   

    def train_model(self,x,y):
        try:
            xgb_clf =  XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score  =f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score  =f1_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)