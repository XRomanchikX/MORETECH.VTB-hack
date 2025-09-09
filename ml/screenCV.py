from collections import OrderedDict
from openai import OpenAI
from modules.extract_data_from_docx import extract_text_from_file, extract_all_text

class InitialScreening:
    def __init__(
                self,
                PathToVacancy: str,
                PathToCV: str,
                ApiUrl: str,
                ApiKey: str,
                mode: int = 0
            ):
        """
        Mode: int

        mode = 0 (default): load files

        mode = 1: load text (give text to Pathes)
        """
        self.PathToVacancy = PathToVacancy
        self.PathToCV = PathToCV
        self.base_url = ApiUrl
        self.api_key = ApiKey

        if mode == 0:
            self.vacancy, self.cv = self.__load_file(self.PathToVacancy), self.__load_file(self.PathToCV)
        else:
            self.vacancy, self.cv = self.PathToVacancy, self.PathToCV

        self.vacancy = self.__formating_text(self.vacancy)
        self.cv = self.__formating_text(self.cv)

    def __load_file(self, file: str):
        data = ""

        if file.split(".")[-1] == "docx":
            data = extract_all_text(file)

        elif file.split(".")[-1] == "rtf":
            data = extract_text_from_file(file)

        else:
            assert FileExistsError("File not found. Please review your path!")

        return data
    
    def __formating_text(self, text: str):
        lines = list(OrderedDict.fromkeys(text.split('\n')))
        
        return ' '.join(lines)
    
    def check_cv(self) -> float:
        system_prompt = f"""
        Analyze the vacancy and the candidate's resume. YOU MUST RETURN ONLY A NUMBER IN THE FORMAT X.X%, WHERE:

        X.X is a percentage from 0.0 to 100.0 with exactly one decimal place;
        Examples: 45.0%, 99.9%, 0.0%.
        STRICTLY PROHIBITED:

        Adding text, explanations, symbols (including spaces, brackets, hyphens);
        Deviating from the format (for example: 45%, 45.0%, 45.0 %, 100% - unacceptable);
        Indicating values ​​outside the range 0.0–100.0.
        IF IT IS IMPOSSIBLE TO ESTIMATE - RETURN 0.0%.
        NO EXCEPTIONS. ONLY A NUMBER IN THE SPECIFIED FORMAT.

        Explanation:

        Clear X.X% template with emphasis on one decimal place and no spaces.
        Strict prohibitions on any deviations (error examples included for contrast).
        Indication of minimum/maximum threshold and behavior under uncertainty.
        Repetition of key requirements to minimize errors.
        No introductory phrases - just the gist, so that the neural network does not add unnecessary things.

        Vacancy requirements:
        {self.vacancy}
        """

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CV: {self.cv}"}
            ],
            temperature=0
        )


        return (float(completion.choices[0].message.content.replace("%", "")))

