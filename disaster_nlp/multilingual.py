"""
Multilingual disaster tweet classification support.
Includes utilities for multilingual evaluation and per-language performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Sample multilingual disaster tweets for evaluation
MULTILINGUAL_DISASTER_TWEETS = {
    'en': [
        {'text': 'massive earthquake strikes city leaving thousands homeless', 'target': 1},
        {'text': 'flooding across three states forces evacuations', 'target': 1},
        {'text': 'just finished my morning coffee and ready for work', 'target': 0},
        {'text': 'beautiful sunset over the mountains today', 'target': 0},
        {'text': 'wildfires rage out of control destroying homes', 'target': 1},
        {'text': 'hurricane warning issued for coastal areas', 'target': 1},
        {'text': 'love spending time with family on weekends', 'target': 0},
        {'text': 'toxic gas leak forces neighborhood evacuation', 'target': 1},
    ],
    'es': [
        {'text': 'terremoto masivo golpea la ciudad dejando miles sin hogar', 'target': 1},
        {'text': 'inundaciones en tres estados fuerzan evacuaciones', 'target': 1},
        {'text': 'acabo de terminar mi café de la mañana', 'target': 0},
        {'text': 'hermosa puesta de sol sobre las montañas hoy', 'target': 0},
        {'text': 'incendios forestales fuera de control destruyen hogares', 'target': 1},
        {'text': 'advertencia de huracán emitida para áreas costeras', 'target': 1},
        {'text': 'me encanta pasar tiempo con la familia los fines de semana', 'target': 0},
        {'text': 'fuga de gas tóxico fuerza evacuación del vecindario', 'target': 1},
    ],
    'fr': [
        {'text': 'tremblement de terre massif frappe la ville laissant des milliers sans abri', 'target': 1},
        {'text': 'inondations dans trois États forcent les évacuations', 'target': 1},
        {'text': 'je viens de terminer mon café du matin', 'target': 0},
        {'text': 'beau coucher de soleil sur les montagnes aujourd hui', 'target': 0},
        {'text': 'les feux de forêt font rage incontrôlés détruisant les maisons', 'target': 1},
        {'text': 'avertissement d ouragan émis pour les zones côtières', 'target': 1},
        {'text': 'j aime passer du temps avec la famille le week end', 'target': 0},
        {'text': 'fuite de gaz toxique force l évacuation du quartier', 'target': 1},
    ],
    'de': [
        {'text': 'massives Erdbeben trifft Stadt und hinterlässt Tausende obdachlos', 'target': 1},
        {'text': 'Überschwemmungen in drei Bundesstaaten erzwingen Evakuierungen', 'target': 1},
        {'text': 'habe gerade meinen Morgenkaffee beendet', 'target': 0},
        {'text': 'wunderschöner Sonnenuntergang über den Bergen heute', 'target': 0},
        {'text': 'Waldbrände außer Kontrolle zerstören Häuser', 'target': 1},
        {'text': 'Hurrikan Warnung für Küstenbereiche ausgegeben', 'target': 1},
        {'text': 'ich liebe es Zeit mit der Familie am Wochenende zu verbringen', 'target': 0},
        {'text': 'giftige Gasleck erzwingt Evakuierung der Nachbarschaft', 'target': 1},
    ],
    'pt': [
        {'text': 'terremoto massivo atinge cidade deixando milhares desabrigados', 'target': 1},
        {'text': 'enchentes em três estados forçam evacuações', 'target': 1},
        {'text': 'acabei de terminar meu café da manhã', 'target': 0},
        {'text': 'pôr do sol lindo sobre as montanhas hoje', 'target': 0},
        {'text': 'incêndios florestais fora de controle destroem casas', 'target': 1},
        {'text': 'aviso de furacão emitido para áreas costeiras', 'target': 1},
        {'text': 'eu adoro passar tempo com a família nos fins de semana', 'target': 0},
        {'text': 'vazamento de gás tóxico força evacuação do bairro', 'target': 1},
    ],
    'ar': [
        {'text': 'زلزال ضخم يضرب المدينة تاركا الآلاف بلا مأوى', 'target': 1},
        {'text': 'الفيضانات في ثلاث ولايات تفرض الإخلاء', 'target': 1},
        {'text': 'انتهيت للتو من قهوة صباحي', 'target': 0},
        {'text': 'غروب جميل على الجبال اليوم', 'target': 0},
        {'text': 'حرائق الغابات خارج السيطرة تدمر المنازل', 'target': 1},
        {'text': 'تحذير من الإعصار يصدر للمناطق الساحلية', 'target': 1},
        {'text': 'أحب قضاء الوقت مع العائلة في عطلات نهاية الأسبوع', 'target': 0},
        {'text': 'تسرب غاز سام يفرض إخلاء الحي', 'target': 1},
    ],
    'hi': [
        {'text': 'भारी भूकंप शहर को प्रभावित करता है हजारों को बेघर छोड़ता है', 'target': 1},
        {'text': 'तीन राज्यों में बाढ़ निकासी को मजबूर करती है', 'target': 1},
        {'text': 'मैंने अभी अपनी सुबह की कॉफी पूरी की है', 'target': 0},
        {'text': 'आज पहाड़ों पर सुंदर सूर्यास्त', 'target': 0},
        {'text': 'जंगल की आग नियंत्रण से बाहर घरों को नष्ट करती है', 'target': 1},
        {'text': 'तूफान चेतावनी तटीय क्षेत्रों के लिए जारी', 'target': 1},
        {'text': 'मुझे सप्ताहांत पर परिवार के साथ समय बिताना पसंद है', 'target': 0},
        {'text': 'जहरीली गैस लीक पड़ोस को खाली करने के लिए मजबूर करता है', 'target': 1},
    ],
    'tr': [
        {'text': 'muazzam deprem şehre çarpıyor binlercesini evsiz bırakıyor', 'target': 1},
        {'text': 'üç eyaletteki seller tahliyeyi zorluyor', 'target': 1},
        {'text': 'sabah kahvemi az önce bitirdim', 'target': 0},
        {'text': 'bugün dağlar üzerinde güzel gün batımı', 'target': 0},
        {'text': 'kontrol dışı orman yangınları evleri yok ediyor', 'target': 1},
        {'text': 'kıyı bölgeleri için kasırga uyarısı verildi', 'target': 1},
        {'text': 'hafta sonları ailemle zaman geçirmeyi seviyorum', 'target': 0},
        {'text': 'zehirli gaz sızıntısı mahallede tahliyeyi zorluyor', 'target': 1},
    ],
}


class MultilingualDataset:
    """Manages multilingual disaster tweet evaluation datasets."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize multilingual dataset.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.language_data = MULTILINGUAL_DISASTER_TWEETS.copy()
    
    def get_language_list(self) -> List[str]:
        """Get list of available languages."""
        return list(self.language_data.keys())
    
    def get_language_texts_and_labels(
        self,
        language: str
    ) -> Tuple[List[str], List[int]]:
        """
        Get texts and labels for a specific language.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Tuple of (texts, labels)
        """
        if language not in self.language_data:
            raise ValueError(f"Language {language} not supported. Available: {self.get_language_list()}")
        
        data = self.language_data[language]
        texts = [item['text'] for item in data]
        labels = [item['target'] for item in data]
        
        return texts, labels
    
    def get_multilingual_dataset(
        self,
        languages: Optional[List[str]] = None
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Get combined dataset across multiple languages.
        
        Args:
            languages: Specific languages to include. None = all.
            
        Returns:
            Tuple of (texts, labels, language_labels)
        """
        if languages is None:
            languages = self.get_language_list()
        
        all_texts = []
        all_labels = []
        all_language_labels = []
        
        for lang in languages:
            texts, labels = self.get_language_texts_and_labels(lang)
            all_texts.extend(texts)
            all_labels.extend(labels)
            all_language_labels.extend([lang] * len(texts))
        
        return all_texts, all_labels, all_language_labels
    
    def get_dataframe(
        self,
        languages: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get multilingual dataset as pandas DataFrame.
        
        Args:
            languages: Specific languages to include. None = all.
            
        Returns:
            DataFrame with columns: text, target, language
        """
        texts, labels, lang_labels = self.get_multilingual_dataset(languages)
        
        return pd.DataFrame({
            'text': texts,
            'target': labels,
            'language': lang_labels
        })
    
    @staticmethod
    def language_name(lang_code: str) -> str:
        """Get human-readable language name."""
        names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'pt': 'Portuguese',
            'it': 'Italian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
        }
        return names.get(lang_code, f'Unknown ({lang_code})')


class PerLanguageEvaluator:
    """Evaluates model performance per language."""
    
    def __init__(self):
        """Initialize per-language evaluator."""
        self.results = defaultdict(lambda: {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'samples': 0
        })
    
    def add_language_results(
        self,
        language: str,
        model_name: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        num_samples: int
    ):
        """
        Add evaluation results for a language.
        
        Args:
            language: Language code
            model_name: Name of the model
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1: F1 score
            num_samples: Number of samples evaluated
        """
        key = f"{language}_{model_name}"
        self.results[key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': num_samples,
            'language': language,
            'model': model_name
        }
    
    def get_language_summary(
        self,
        language: str
    ) -> Dict[str, float]:
        """
        Get summary of results for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with average metrics across all models for this language
        """
        lang_results = [
            v for k, v in self.results.items()
            if k.startswith(language + '_')
        ]
        
        if not lang_results:
            return {}
        
        return {
            'language': language,
            'num_models': len(lang_results),
            'avg_accuracy': np.mean([r['accuracy'] for r in lang_results]),
            'avg_precision': np.mean([r['precision'] for r in lang_results]),
            'avg_recall': np.mean([r['recall'] for r in lang_results]),
            'avg_f1': np.mean([r['f1'] for r in lang_results]),
            'total_samples': sum([r['samples'] for r in lang_results])
        }
    
    def get_all_results_dataframe(self) -> pd.DataFrame:
        """Get all results as DataFrame."""
        return pd.DataFrame(list(self.results.values()))
    
    def get_per_language_summary(self) -> pd.DataFrame:
        """
        Get per-language summary across all models.
        
        Returns:
            DataFrame with language-aggregated metrics
        """
        languages = set(k.split('_')[0] for k in self.results.keys())
        summaries = [self.get_language_summary(lang) for lang in sorted(languages)]
        return pd.DataFrame(summaries)
    
    def generate_language_report(self) -> str:
        """
        Generate markdown report of per-language performance.
        
        Returns:
            Markdown formatted report
        """
        summary_df = self.get_per_language_summary()
        
        if summary_df.empty:
            return "No per-language results available."
        
        report = "## Per-Language Performance Report\n\n"
        report += "### Language Summary\n\n"
        report += summary_df.to_markdown(index=False)
        report += "\n\n"
        
        report += "### Language Details\n\n"
        for _, row in summary_df.iterrows():
            lang = row['language']
            lang_name = MultilingualDataset.language_name(lang)
            report += f"#### {lang_name} ({lang})\n"
            report += f"- Accuracy: {row['avg_accuracy']:.4f}\n"
            report += f"- Precision: {row['avg_precision']:.4f}\n"
            report += f"- Recall: {row['avg_recall']:.4f}\n"
            report += f"- F1 Score: {row['avg_f1']:.4f}\n"
            report += f"- Samples: {int(row['total_samples'])}\n\n"
        
        return report
