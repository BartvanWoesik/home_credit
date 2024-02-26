import hydra
from model.modelorchastrator import ModelOrchestrator


@hydra.main(config_path="", config_name="config")
def create_model_orchestrator(cfg):
    model_orchestrator = ModelOrchestrator(cfg=cfg.model)
    return model_orchestrator


model_orchestrator = create_model_orchestrator()
