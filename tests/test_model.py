import torch

from bifusion.model import BiFusionYOLO, count_parameters


def test_deployed_parameter_count():
    model = BiFusionYOLO()
    report = model.parameter_report()
    assert report["backbone"] == 1_139_680
    assert report["neck"] == 150_656
    assert report["heads"] == 492_774
    assert report["total"] == 1_783_110


def test_training_model_is_larger_than_deployed():
    model = BiFusionYOLO()
    assert count_parameters(model) > count_parameters(model.deploy())


def test_forward_shapes():
    model = BiFusionYOLO().eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 640, 640))
    assert len(outputs) == 3
    assert [tuple(o.shape[1:]) for o in outputs] == [
        (65, 160, 160),
        (65, 80, 80),
        (65, 40, 40),
    ]


def test_deploy_is_idempotent_in_shape():
    model = BiFusionYOLO().eval()
    deployed = model.deploy()
    with torch.no_grad():
        a = model(torch.zeros(1, 3, 320, 320))
        b = deployed(torch.zeros(1, 3, 320, 320))
    assert all(x.shape == y.shape for x, y in zip(a, b))


def test_deploy_matches_training_output():
    torch.manual_seed(0)
    model = BiFusionYOLO().eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        a = model(x)
        b = model.deploy()(x)
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, b))
