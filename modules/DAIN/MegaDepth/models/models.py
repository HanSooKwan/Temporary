
def create_model(opt,pretrained=None):
    model = None
    from modules.DAIN.MegaDepth.models.HG_model import HGModel
    model = HGModel(opt,pretrained)
    # print("model [%s] was created" % (model.name()))
    return model
