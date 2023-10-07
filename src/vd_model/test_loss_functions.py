import src.vd_model.loss_functions as loss_func
import numpy as np

def test_kolomogorow_smirnow():
    func = loss_func.kolomogorow_smirnow
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0, 1.1, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0, 1.2, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0.1, 1, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0.2, 1, 1000), np.random.normal(0, 1, 1000))

    func = loss_func.kl_divergence
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0, 1.1, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0, 1.2, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0.1, 1, 1000), np.random.normal(0, 1, 1000))
    assert func(np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)) < func(np.random.normal(0.2, 1, 1000), np.random.normal(0, 1, 1000))
     

    assert False