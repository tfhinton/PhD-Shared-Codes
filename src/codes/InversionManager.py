import pickle

class InversionManager:
    def __init__(self, InversionClass, inversion_args, run_labels, picklepath=None):
        self.Inversion = InversionClass
        self.inversion_args = inversion_args
        self.run_labels = run_labels
        self.results = {}
        self.picklepath = picklepath
    
    def run(self):
        for args, label in zip(self.inversion_args, self.run_labels):
            print("Label:", label)
            print("Inversion args:", args)
            print(f"\n\n\n########   INVERSION RUN: {label}  ########\n\n\n")
            inversion = self.Inversion(*args)
            inversion = inversion.run()
            self.results[label] = inversion.result
            if self.picklepath is not None:
                with open(self.picklepath, "wb") as f:
                    pickle.dump(self, f)
        
        return self