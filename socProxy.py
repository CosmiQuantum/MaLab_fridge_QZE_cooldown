import Pyro4
from qick import QickConfig

def makeProxy(ns_host='192.168.1.199', ns_port=8888, proxy_name="myqick"): #'192.168.1.199'
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    for k, v in ns.list().items():
        print(k, v)




    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())
    print(soccfg)
    return(soc,soccfg)