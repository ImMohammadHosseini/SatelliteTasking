from Basilisk.moduleTemplates import cModuleTemplate
from Basilisk.moduleTemplates import cppModuleTemplate
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.architecture import messaging


from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros

from Basilisk.utilities import orbitalMotion, macros, unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk

import matplotlib.pyplot as plt
from Basilisk.utilities import unitTestSupport
import sys
def run():
	scSim = SimulationBaseClass.SimBaseClass()

	#dynProcess = scSim.CreateNewProcess("name", priority)
	dynProcess = scSim.CreateNewProcess("dynamicsProcess")
	fswProcess = scSim.CreateNewProcess("fswProcess")

	#dynProcess.addTask(scSim.CreateNewTask("name", updateRate, priority))
	#dynProcess.addTask(scSim.CreateNewTask("name", updateRate, priority, FirstStart=delayStartTime))

	dynProcess.addTask(scSim.CreateNewTask("dynamicsTask", macros.sec2nano(5.)))

	mod1 = cModuleTemplate.cModuleTemplate()
	mod1.ModelTag = "cModule1"

	mod2 = cppModuleTemplate.CppModuleTemplate()
	mod2.ModelTag = "cppModule2"

	mod3 = cModuleTemplate.cModuleTemplate()
	mod3.ModelTag = "cModule3"
	
	#scSim.AddModelToTask("taskName", module, priority)
	scSim.AddModelToTask("dynamicsTask", mod1)
	scSim.AddModelToTask("dynamicsTask", mod2, 10)
	scSim.AddModelToTask("dynamicsTask", mod3, 5)
	
	mod1.dataInMsg.subscribeTo(mod1.dataOutMsg)
	msgData = messaging.CModuleTemplateMsgPayload()
	msgData.dataVector = [1., 2., 3.]
	msg = messaging.CModuleTemplateMsg().write(msgData)
	
	# connect to stand-alone msg
	mod1.dataInMsg.subscribeTo(msg)

	msgRec = mod1.dataOutMsg.recorder()
	scSim.AddModelToTask("dynamicsTask", msgRec)
	msgRec2 = mod1.dataOutMsg.recorder(macros.sec2nano(20.))
	scSim.AddModelToTask("dynamicsTask", msgRec2)

	dynProcess.addTask(scSim.CreateNewTask("sensorTask", macros.sec2nano(10.)))
	fswProcess.addTask(scSim.CreateNewTask("fswTask", macros.sec2nano(10.)))

	scSim.InitializeSimulation()
	scSim.ConfigureStopTime(macros.sec2nano(20.0))
	scSim.ExecuteSimulation()

	scSim.ShowExecutionOrder()
	fig = scSim.ShowExecutionFigure(True)

	# plot recorded data
	plt.close("all")
	plt.figure(1)
	figureList = {}
	for idx in range(3):
		plt.plot(msgRec.times() * macros.NANO2SEC, msgRec.dataVector[:, idx],
			color=unitTestSupport.getLineColor(idx, 3),
			label='$x_{' + str(idx) + '}$')
		plt.plot(msgRec2.times() * macros.NANO2SEC, msgRec2.dataVector[:, idx],
			'--',
			color=unitTestSupport.getLineColor(idx, 3),
			label=r'$\hat x_{' + str(idx) + '}$')
	plt.legend(loc='lower right')
	plt.xlabel('Time [sec]')
	plt.ylabel('Module Data [units]')
	#if "pytest" not in sys.modules:
	plt.show()
	figureList["bsk-4"] = plt.figure(1)
	plt.close("all")
	return figureList
if __name__ == "__main__":
	run()