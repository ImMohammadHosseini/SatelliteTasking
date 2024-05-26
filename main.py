from Basilisk.moduleTemplates import cModuleTemplate
from Basilisk.moduleTemplates import cppModuleTemplate
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros


from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros

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

	dynProcess.addTask(scSim.CreateNewTask("sensorTask", macros.sec2nano(10.)))
	fswProcess.addTask(scSim.CreateNewTask("fswTask", macros.sec2nano(10.)))

	scSim.InitializeSimulation()
	scSim.ConfigureStopTime(macros.sec2nano(20.0))
	scSim.ExecuteSimulation()

	scSim.ShowExecutionOrder()
	fig = scSim.ShowExecutionFigure(True)
	return
if __name__ == "__main__":
	run()