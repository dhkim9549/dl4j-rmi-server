package com.example.rmitest;

// Java program to implement the Search interface
import java.rmi.*;
import java.rmi.server.*;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import com.google.gson.Gson;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SearchQuery extends UnicastRemoteObject implements Search {

	private static final Logger log = LoggerFactory.getLogger(SearchQuery.class);
	MultiLayerNetwork model = null;
	MultiLayerNetwork model2 = null;

        protected SearchQuery() throws RemoteException {
                super();
                try {
                        model = readModelFromFile("/root/data/xor-nn-model.zip");
			model2 = readModelFromFile("/root/data/xor-nn-model-2.zip");
                } catch (Exception e) {
                        e.printStackTrace();
                }
        }

        public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

                log.info("Deserializing model...");

                File locationToSave = new File(fileName);
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

                log.info("Deserializing model complete.");

                return model;
        }

	// Implementation of the query interface
	public String query(String search) throws RemoteException {
	
		log.info("search = " + search);

		String result = "";

		try { 

			Gson gson = new Gson();
			HashMap map = gson.fromJson(search, HashMap.class);
			ArrayList<String> list = (ArrayList<String>)map.get("inputList"); 
			String nnType = (String)map.get("nnType");

			INDArray inputA = Nd4j.zeros(1, 2);
			for(int i = 0; i < list.size(); i++) {
				inputA.putScalar(new int[]{0, i}, Double.parseDouble(list.get(i)));
			}

			INDArray outputA = null;
			if(nnType.equals("1")) {
				outputA = model.output(inputA);
			} else if(nnType.equals("2")) {
				outputA = model2.output(inputA);
			}

			result = "" + outputA;

		} catch(Exception e) {
			e.printStackTrace();
			throw e;
		}

		log.info("result = " + result);

		return result;
	}
}

