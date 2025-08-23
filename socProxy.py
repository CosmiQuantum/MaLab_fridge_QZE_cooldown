import Pyro4
from qick import QickConfig

def makeProxy(ns_host='http://192.168.1.199/', ns_port='8888', proxy_name="qickbox"):
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    #ns_host = "rfsoc216-loud1.dhcp.fnal.gov" # loud1 is the QICK RF board, loud2 is the "old QICK" board

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    soc = Pyro4.Proxy(ns.lookup(proxy_name))
    soccfg = QickConfig(soc.get_cfg())
    print(soccfg)
    return(soc,soccfg)