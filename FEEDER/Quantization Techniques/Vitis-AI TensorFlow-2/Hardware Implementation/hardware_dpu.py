import numpy as np
import vart
import xir
import time

# Φόρτωση του υπογραφήματος από το xmodel
def load_graph(xmodel_file):
    return xir.Graph.deserialize(xmodel_file)

def run_model(model_file, input_data):
    # Φόρτωση του υπογραφήματος
    graph = load_graph(model_file)
    root_subgraph = graph.get_root_subgraph()

    subgraph = None
    for sub in root_subgraph.toposort_child_subgraph():
        if sub.has_attr("device") and sub.get_attr("device").upper() == "DPU":
            subgraph = sub
            break
        
    if subgraph is None:
        raise ValueError("Δεν βρέθηκε υπογράφημα DPU στο μοντέλο.")
    
    dpu_runner = vart.Runner.create_runner(subgraph, "run")

    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()

    input_fixpos = input_tensors[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    output_fixpos = output_tensors[0].get_attr("fix_point")
    output_scale = 1 / (2 ** output_fixpos)

    binary_outputs = []

    time_start = time.time()

    for i in range(len(input_data)):
    
        # Eικόνα (1, 32, 32, 9)
        single_input = input_data[i:i+1]

        # Προετοιμασία δεδομένων εισόδου
        input_data_quantized = (single_input * input_scale).astype(np.int8)
        input_tensor_buffer = np.zeros(input_tensors[0].dims, dtype=np.int8)
        input_tensor_buffer[:] = input_data_quantized

        # Εκτέλεση του υπογραφήματος
        output_tensor_buffer = np.zeros(output_tensors[0].dims, dtype=np.int8)
        job_id = dpu_runner.execute_async([input_tensor_buffer], [output_tensor_buffer])
        dpu_runner.wait(job_id)

        # Απο-κβαντισμός των εξόδων
        output_data_dequantized = output_tensor_buffer.astype(np.float64) * output_scale

        # Softmax
        max_val = np.max(output_data_dequantized)
        stable_softmax_output = np.exp(output_data_dequantized - max_val)
        softmax_output = stable_softmax_output / np.sum(stable_softmax_output)

        # Μετατροπή σε δυαδικό πίνακα [0, 1, 0, 0]
        binary_output = (softmax_output == np.max(softmax_output)).astype(int)
        
        # Αποθήκευση του αποτελέσματος
        binary_outputs.append(binary_output)

    #Υπολογισμός χρόνου εκτέλεσης
    time_end = time.time()
    timetotal = time_end - time_start
    print("Total Time:", timetotal)

    return np.squeeze(np.array(binary_outputs))

def calculate_accuracy(predictions, ground_truth):
    # Σύγκριση προβλέψεων με το πραγματικό output
    correct_predictions = np.sum(np.all(predictions == ground_truth, axis=1))
    accuracy = correct_predictions / len(ground_truth)
    return accuracy

# Φόρτωση του x_tst
x_tst = np.load('x_tst_file.npy')
y_tst = np.load('y_tst_file.npy')

# Εκτέλεση του μοντέλου 
model_file = r'/home/root/for_fpga/quantized_hardware.xmodel'
output_data = run_model(model_file, x_tst[0:100])

print("Real Output", y_tst[0:100])

# Επεξεργασία των αποτελεσμάτων
print("FPGA Output Data:", output_data)

accuracy = calculate_accuracy(output_data, y_tst[0:100])
print(f"Accuracy: {accuracy * 100:.2f}%")


