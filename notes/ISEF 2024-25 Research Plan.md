**Name:** Sumer Chaudhary  
**Grade:** 9th  
**Title:** Assessing Effect of Change in Computational Complexity (Space and Time Variables) on Accuracy And Efficiency Of CSS Codes, Applied on Bell States

**Rationale**  
One of the major uses for quantum computing is optimization of systems. One such system that could be optimized is Air Traffic Control, which has to manage thousands of flights per day. However, as a report from the United States Government Accountability Office found, “of \[Air Traffic Control’s\] 138 systems, 51 (37 percent) were unsustainable and 54 (39 percent) were potentially unsustainable” (Air Traffic Control 2). These antiquated systems are already causing accidents, such as the tragic accident of American Airlines flight 5342 on January 29, 2025\. This accident is a direct result of the outdated Air Traffic Control systems, and they need an overhaul. Quantum computing would be one of the best methods for Air Traffic Control, as it can handle the increasing traffic with ease and efficiently model the optimal path for every flight. However, this system would need to be as precise and accurate as possible, requiring the fragility of quantum computing be taken care of. The fragility of quantum computers mainly lies in that they “are highly susceptible to unwanted environmental interactions and, therefore, error” (Massari, 2024). Any of these “unwanted environmental interactions,” such as heat, causes the quantum circuit to become slightly more erroneous (Massari, 2024). To help with the error, most quantum computers today run a single circuit thousands of times, and average the result. However, this is ridiculously inefficient, which is why I focused on quantum error correction, so results are more accurate and precise. 

**Research Question:** How do different quantum error codes handle trade offs between the accuracy and the time complexity of the program?

**List of materials**

- Computer with a Python-based IDE with the libraries qiskit, numpy, and matplotlib installed, with at least 2 GB of RAM  
- Open Account or higher on the IBM Quantum Platform (Access to quantum computers with 127 qubits)

**Procedure**

1) In a Python-based IDE, make the following functions:  
   1) create\_bell\_state(): returns a quantum circuit prepared in a bell state  
   2) encode\_with\_shors(): takes in a quantum circuit and encodes it with Shor’s Code  
   3) encode\_with\_shors(): takes in a quantum circuit encoded with Shor’s Code and measures its stabilizers  
   4) shor\_correct\_errors(): takes in a quantum circuit encoded with Shor’s Code and its syndrome and corrects its errors  
   5) decode\_with\_shors(): takes in a quantum circuit encoded with Shor’s Code and decodes it into one qubit  
   6) create\_shor\_bell\_state(): returns a quantum circuit, prepared in a bell state, encoded, error-corrected, and decoded with Shor’s Code  
   7) encode\_with\_steane(): takes in a quantum circuit and encodes it with Steane’s Code  
   8) steane\_measure\_syndrome(): takes in a quantum circuit encoded with Steane’s Code and measures its stabilizers  
   9) steane\_correct\_errors(): takes in a quantum circuit encoded with Steane’s Code and its syndrome and corrects its errors  
   10) decode\_with\_steanes(): takes in a quantum circuit encoded with Steane’s Code and decodes it into one qubit  
   11) create\_steane\_bell\_state(): returns a quantum circuit, prepared in a bell state, encoded, error-corrected, and decoded with Steane’s Code  
   12) encode\_with\_fqc(): takes in a quantum circuit and encodes it with The Five-Qubit Code  
   13) fqc\_measure\_syndrome(): takes in a quantum circuit encoded with the Five-Qubit Code and measures its stabilizers  
   14) fqc\_correct\_errors(): takes in a quantum circuit encoded with the Five-Qubit Code and its syndrome and corrects its errors  
   15) decode\_with\_fqcs(): takes in a quantum circuit encoded with the Five-Qubit Code and decodes it into one qubit  
   16) create\_fqc\_bell\_state(): returns a quantum circuit, prepared in a bell state, encoded, error-corrected, and decoded with the Five-Qubit Code  
2) Run create\_bell\_state(), create\_shor\_bell\_state(), create\_steane\_bell\_state(), and create\_fqc\_bell\_state(), and save the results as a list of circuits circuits \= \[control\_qc, shor\_qc, steane\_qc, and fqc\_qc\], respectively.  
3) Through qiskit\_ibm\_runtime, QiskitRuntimeService, access the ibm\_sherbrooke QPU and save it as qBackend  
4) Generate a passmanager for the ibm\_sherbrooke QPU and create a list of ISA-mapped circuits by running each circuit in circuits through the passmanager  
5) Run the ISA-mapped circuits on the ibm\_sherbrooke QPU by using a Sampler  
6) Save the PUB\_results of each job for post-processing.

**Risk and Safety**  
No risks present. No safety precautions necessary.

**Data Analysis**  
Create four dictionaries: results, errors, times, depths, and num\_qubits. The keys for these dictionaries should be the names of the circuits. Iterate through the PUB\_results, and, for each circuit save the raw results to results, save the number of 01 and 10 results to errors, save the time spent on the QPU to times, save the circuit’s depth to depths, and save the number of qubits in the circuit to num\_qubits. Plot these dictionaries alone as a histogram using matplotlib.pyplot.bar, and against each other as a graph using matplotlib.pyplot.plot.

**Bibliography**  
Masssari, Paul. “At the Quantum Frontier | the Harvard Kenneth C. Griffin Graduate School of Arts and Sciences.” Harvard.edu, 2024, gsas.harvard.edu/news/quantum-frontier. Accessed 5 Dec. 2024\.  
“IBM Quantum.” IBM Quantum, quantum.ibm.com/.  
‌Preskill‌, John. “Quantum Computing (CST Part II) Lecture 13: Quantum Error Correction.”   
Roffe, Joschka. “Quantum Error Correction: An Introductory Guide.” Contemporary Physics, vol. 60, no. 3, 3 July 2019, pp. 226–245, arxiv.org/pdf/1907.11157.pdf, https://doi.org/10.1080/00107514.2019.1667078.‌  
Air Traffic Control: FAA Actions Are Urgently Needed to Modernize Aging Systems. United States Government Accountability Office, Sep 2024, https://www.gao.gov/assets/gao-24-107001.pdf?ref=tippinsights.com. Accessed 9 Feb. 2024  
Wikipedia Contributors. “Quantum Error Correction.” Wikipedia, Wikimedia Foundation, 3 Dec. 2024\.  
