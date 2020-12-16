import glob
import os

class BookscorpusTextFormatting:
    def __init__(self, books_path, output_filename, recursive = False, interval = 500):
        self.books_path = books_path
        self.recursive = recursive
        self.output_filename = output_filename.split('.')
        self.interval = interval

    # This puts one book per line

    def merge(self):
        count = 0
        for filename in glob.glob(self.books_path + '/' + '*.txt', recursive=True):
            if count == 0:
                ofile_name = '.'.join([self.output_filename[0]+'-'+str(count//500), self.output_filename[1]])
                ofile = open(ofile_name, mode='w', encoding='utf-8-sig', newline='\n')
            elif count%self.interval == 0:
                print(count)
                ofile.close()
                ofile_name = '.'.join([self.output_filename[0]+'-'+str(count//500), self.output_filename[1]])
                ofile = open(ofile_name, mode='w', encoding='utf-8-sig', newline='\n')
            file = open(filename, mode='r', encoding='utf-8-sig', newline='\n')
            for line in file:
                if line.strip() != '':
                    ofile.write(line.strip() + ' ')
            ofile.write("\n\n")
            count += 1
        ofile.close()

data_dir = 'BookCorpus/books1/epubtxt/'
output_name_format = 'BookCorpus/after_prepare/bookcorpus.txt'

FormatTool = BookscorpusTextFormatting(data_dir, output_name_format)
FormatTool.merge()


